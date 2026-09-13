using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Server
{
    [JsonSerializable(typeof(float[]))]
    internal partial class InferenceJsonContext : JsonSerializerContext
    {
    }

    /// <summary>
    /// Ultra-lightweight in-process HTTP/REST Micro-Server (&lt;1ms response latency) for embedded AI serving.
    /// </summary>
    public sealed class EmbeddedServer : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly DynamicBatcher _batcher;
        private readonly HttpListener _listener;
        private readonly CancellationTokenSource _cts;
        private readonly Task _listenTask;
        private long _totalRequests;
        private long _totalLatencyMicroseconds;
        private bool _disposed;

        public EmbeddedServer(InferenceSession session, int port = 8080)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _batcher = new DynamicBatcher(session);

            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://localhost:{port}/");
            _listener.Prefixes.Add($"http://127.0.0.1:{port}/");

            _cts = new CancellationTokenSource();
            _listener.Start();
            _listenTask = Task.Run(ListenLoopAsync);
        }

        public long TotalRequests => Interlocked.Read(ref _totalRequests);
        public double AverageLatencyMicroseconds
        {
            get
            {
                long req = TotalRequests;
                return req > 0 ? (double)Interlocked.Read(ref _totalLatencyMicroseconds) / req : 0;
            }
        }

        private async Task ListenLoopAsync()
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                try
                {
                    var context = await _listener.GetContextAsync();
                    _ = Task.Run(() => HandleRequestAsync(context));
                }
                catch (HttpListenerException) when (_cts.IsCancellationRequested)
                {
                    break;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (Exception)
                {
                    // Continue listening loop
                }
            }
        }

        private async Task HandleRequestAsync(HttpListenerContext context)
        {
            var req = context.Request;
            var res = context.Response;
            var sw = Stopwatch.StartNew();

            try
            {
                if (req.HttpMethod == "GET" && req.Url?.AbsolutePath == "/health")
                {
                    byte[] payload = Encoding.UTF8.GetBytes("{\"status\":\"healthy\",\"engine\":\"TokenVector.Inference\"}");
                    res.ContentType = "application/json";
                    res.StatusCode = 200;
                    await res.OutputStream.WriteAsync(payload);
                    return;
                }

                if (req.HttpMethod == "GET" && req.Url?.AbsolutePath == "/metrics")
                {
                    string metrics = $"{{\"total_requests\":{TotalRequests},\"avg_latency_us\":{AverageLatencyMicroseconds:F2}}}";
                    byte[] payload = Encoding.UTF8.GetBytes(metrics);
                    res.ContentType = "application/json";
                    res.StatusCode = 200;
                    await res.OutputStream.WriteAsync(payload);
                    return;
                }

                if (req.HttpMethod == "POST" && req.Url?.AbsolutePath == "/predict")
                {
                    using var reader = new StreamReader(req.InputStream, Encoding.UTF8);
                    string body = await reader.ReadToEndAsync();
                    float[]? inputData = JsonSerializer.Deserialize(body, InferenceJsonContext.Default.SingleArray);

                    if (inputData != null && inputData.Length > 0)
                    {
                        var inArray = new NDArray<float>(new int[] { 1, inputData.Length });
                        inputData.AsSpan().CopyTo(inArray.AsSpan());

                        var outArray = await _batcher.EnqueueAsync(inArray);

                        string responseJson = JsonSerializer.Serialize(outArray.AsSpan().ToArray(), InferenceJsonContext.Default.SingleArray);
                        byte[] payload = Encoding.UTF8.GetBytes(responseJson);

                        res.ContentType = "application/json";
                        res.StatusCode = 200;
                        await res.OutputStream.WriteAsync(payload);
                    }
                    else
                    {
                        res.StatusCode = 400;
                    }
                }
                else
                {
                    res.StatusCode = 404;
                }
            }
            catch (Exception ex)
            {
                res.StatusCode = 500;
                byte[] errorBytes = Encoding.UTF8.GetBytes($"{{\"error\":\"{ex.Message}\"}}");
                await res.OutputStream.WriteAsync(errorBytes);
            }
            finally
            {
                sw.Stop();
                Interlocked.Increment(ref _totalRequests);
                Interlocked.Add(ref _totalLatencyMicroseconds, (long)(sw.Elapsed.TotalMicroseconds));
                res.Close();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _cts.Cancel();
                try
                {
                    _listener.Stop();
                    _listener.Close();
                }
                catch { }

                _batcher.Dispose();
                _cts.Dispose();
                _disposed = true;
            }
        }
    }
}
