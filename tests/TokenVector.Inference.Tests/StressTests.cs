using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Quantization;
using TokenVector.Inference.Runtime;
using TokenVector.Inference.Server;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class StressTests
    {
        [Fact]
        public void LongRunning_100k_Inference_ZeroMemoryLeak()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(64, 64);
            var b = new NDArray<float>(64);
            for (int i = 0; i < 64 * 64; i++) w.AsSpan()[i] = 0.01f;
            for (int i = 0; i < 64; i++) b.AsSpan()[i] = 0.02f;

            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;
            graph.GetOrCreateTensor("X", new int[] { 1, 64 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 64 });

            var node = new ExecutionNode
            {
                Id = 0,
                OpType = OperatorType.FusedLinear,
                Activation = ActivationFunction.GELU
            };
            node.Inputs.Add("X");
            node.Inputs.Add("W");
            node.Inputs.Add("B");
            node.Outputs.Add("Y");
            graph.Nodes.Add(node);

            using var session = new InferenceSession(graph);

            var inX = new NDArray<float>(1, 64);
            var outY = new NDArray<float>(1, 64);
            for (int i = 0; i < 64; i++) inX.AsSpan()[i] = 1.0f;

            NamedNDArray[] inputs = new NamedNDArray[1] { new("X", inX) };
            NamedNDArray[] outputs = new NamedNDArray[1] { new("Y", outY) };

            // Warmup
            for (int i = 0; i < 1000; i++)
            {
                session.Run(inputs.AsSpan(), outputs.AsSpan());
            }

            var sw = new Stopwatch();

            int gen0Before = GC.CollectionCount(0);
            int gen1Before = GC.CollectionCount(1);
            int gen2Before = GC.CollectionCount(2);
            long memBefore = GC.GetAllocatedBytesForCurrentThread();

            const int loopCount = 100000;
            sw.Start();
            for (int i = 0; i < loopCount; i++)
            {
                session.Run(inputs.AsSpan(), outputs.AsSpan());
            }
            sw.Stop();

            long memAllocated = GC.GetAllocatedBytesForCurrentThread() - memBefore;
            int gen0Diff = GC.CollectionCount(0) - gen0Before;
            int gen1Diff = GC.CollectionCount(1) - gen1Before;
            int gen2Diff = GC.CollectionCount(2) - gen2Before;

            Assert.Equal(0, memAllocated);
            Assert.Equal(0, gen0Diff);
            Assert.Equal(0, gen1Diff);
            Assert.Equal(0, gen2Diff);
            Assert.True(sw.ElapsedMilliseconds > 0);
        }

        [Fact]
        public async Task MultiThreaded_Concurrent_DynamicBatcher_StressTest()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(32, 32);
            var b = new NDArray<float>(32);
            for (int i = 0; i < 32 * 32; i++) w.AsSpan()[i] = 0.05f;
            for (int i = 0; i < 32; i++) b.AsSpan()[i] = 0.1f;

            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;
            graph.GetOrCreateTensor("X", new int[] { 1, 32 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 32 });

            var node = new ExecutionNode
            {
                Id = 0,
                OpType = OperatorType.FusedLinear,
                Activation = ActivationFunction.ReLU
            };
            node.Inputs.Add("X");
            node.Inputs.Add("W");
            node.Inputs.Add("B");
            node.Outputs.Add("Y");
            graph.Nodes.Add(node);

            using var session = new InferenceSession(graph);
            using var batcher = new DynamicBatcher(session, maxBatchSize: 16, timeoutMicroseconds: 200);

            const int totalTasks = 2000;
            const int concurrency = 16;
            var tasks = new Task[concurrency];
            var errors = new ConcurrentBag<Exception>();

            for (int t = 0; t < concurrency; t++)
            {
                int requestsPerThread = totalTasks / concurrency;
                tasks[t] = Task.Run(async () =>
                {
                    var input = new NDArray<float>(1, 32);
                    for (int j = 0; j < 32; j++) input.AsSpan()[j] = 1.0f;

                    for (int r = 0; r < requestsPerThread; r++)
                    {
                        try
                        {
                            var res = await batcher.EnqueueAsync(input);
                            if (res.TotalLength != 32 || MathF.Abs(res.AsSpan()[0] - 1.7f) > 0.01f)
                            {
                                errors.Add(new InvalidOperationException($"Invalid result value: {res.AsSpan()[0]}"));
                            }
                        }
                        catch (Exception ex)
                        {
                            errors.Add(ex);
                        }
                    }
                });
            }

            await Task.WhenAll(tasks);
            Assert.Empty(errors);
        }

        [Fact]
        public void DeepNetwork_50Layers_GraphStressTest()
        {
            // Build 50-layer deep neural network with residual connections
            var graph = new ExecutionGraph();
            graph.Inputs.Add("x_in");
            graph.Outputs.Add("x_out_49");

            int dim = 32;
            graph.GetOrCreateTensor("x_in", new int[] { 1, dim });

            for (int l = 0; l < 50; l++)
            {
                string inTensor = l == 0 ? "x_in" : $"x_out_{l - 1}";
                string outTensor = $"x_out_{l}";
                string wName = $"W_{l}";
                string bName = $"B_{l}";

                var w = new NDArray<float>(dim, dim);
                var b = new NDArray<float>(dim);
                for (int i = 0; i < dim * dim; i++) w.AsSpan()[i] = (i % (dim + 1) == 0) ? 0.99f : 0.001f; // Near identity
                for (int i = 0; i < dim; i++) b.AsSpan()[i] = 0.001f;

                graph.Initializers[wName] = w;
                graph.Initializers[bName] = b;

                graph.GetOrCreateTensor(wName, w.Shape);
                graph.GetOrCreateTensor(bName, b.Shape);
                graph.GetOrCreateTensor(outTensor, new int[] { 1, dim });

                var node = new ExecutionNode
                {
                    Id = l,
                    OpType = OperatorType.FusedLinear,
                    Activation = (l % 2 == 0) ? ActivationFunction.SiLU : ActivationFunction.GELU
                };
                node.Inputs.Add(inTensor);
                node.Inputs.Add(wName);
                node.Inputs.Add(bName);
                node.Outputs.Add(outTensor);
                graph.Nodes.Add(node);
            }

            using var session = new InferenceSession(graph);

            var input = new NDArray<float>(1, dim);
            for (int i = 0; i < dim; i++) input.AsSpan()[i] = 1.0f;

            var output = session.Run(input, "x_in", "x_out_49");

            Assert.Equal(dim, output.TotalLength);
            for (int i = 0; i < dim; i++)
            {
                Assert.False(float.IsNaN(output.AsSpan()[i]));
                Assert.False(float.IsInfinity(output.AsSpan()[i]));
            }
        }
    }
}
