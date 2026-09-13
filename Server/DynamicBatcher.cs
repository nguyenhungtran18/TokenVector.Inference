using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Server
{
    public sealed class BatchRequest
    {
        public NDArray<float> Input { get; }
        public TaskCompletionSource<NDArray<float>> CompletionSource { get; }

        public BatchRequest(NDArray<float> input)
        {
            Input = input;
            CompletionSource = new TaskCompletionSource<NDArray<float>>(TaskCreationOptions.RunContinuationsAsynchronously);
        }
    }

    /// <summary>
    /// Lock-Free Micro-Batching Engine using System.Threading.Channels and spin-wait timeout window (100 - 500 microseconds).
    /// </summary>
    public sealed class DynamicBatcher : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _maxBatchSize;
        private readonly TimeSpan _batchTimeout;
        private readonly Channel<BatchRequest> _channel;
        private readonly CancellationTokenSource _cts;
        private readonly Task _workerTask;
        private bool _disposed;

        public DynamicBatcher(InferenceSession session, int maxBatchSize = 32, int timeoutMicroseconds = 200)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _maxBatchSize = Math.Max(1, maxBatchSize);
            _batchTimeout = TimeSpan.FromMicroseconds(timeoutMicroseconds);

            var options = new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false
            };
            _channel = Channel.CreateUnbounded<BatchRequest>(options);
            _cts = new CancellationTokenSource();
            _workerTask = Task.Run(ProcessBatchesAsync);
        }

        public async Task<NDArray<float>> EnqueueAsync(NDArray<float> input, CancellationToken cancellationToken = default)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(DynamicBatcher));

            var request = new BatchRequest(input);
            await _channel.Writer.WriteAsync(request, cancellationToken);
            return await request.CompletionSource.Task;
        }

        private async Task ProcessBatchesAsync()
        {
            var batch = new List<BatchRequest>(_maxBatchSize);
            var reader = _channel.Reader;

            while (!_cts.Token.IsCancellationRequested)
            {
                batch.Clear();

                try
                {
                    // Wait for at least one item
                    if (await reader.WaitToReadAsync(_cts.Token))
                    {
                        var sw = Stopwatch.StartNew();

                        while (batch.Count < _maxBatchSize && (batch.Count == 0 || sw.Elapsed < _batchTimeout))
                        {
                            if (reader.TryRead(out var req))
                            {
                                batch.Add(req);
                            }
                            else
                            {
                                // Micro-yield / spinwait
                                Thread.SpinWait(10);
                            }
                        }

                        if (batch.Count > 0)
                        {
                            ExecuteBatch(batch);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    foreach (var req in batch)
                        req.CompletionSource.TrySetException(ex);
                }
            }
        }

        private void ExecuteBatch(List<BatchRequest> batch)
        {
            for (int i = 0; i < batch.Count; i++)
            {
                var req = batch[i];
                try
                {
                    var result = _session.Run(req.Input);
                    req.CompletionSource.TrySetResult(result);
                }
                catch (Exception ex)
                {
                    req.CompletionSource.TrySetException(ex);
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _channel.Writer.TryComplete();
                _cts.Cancel();
                try
                {
                    _workerTask.Wait(500);
                }
                catch { }

                _cts.Dispose();
                _disposed = true;
            }
        }
    }
}
