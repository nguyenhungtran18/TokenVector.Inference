using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Inference.Server;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class DynamicBatcherTests
    {
        [Fact]
        public async Task DynamicBatcher_HandlesConcurrentRequestsCorrectly()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");

            var w = new NDArray<float>(4, 4);
            var b = new NDArray<float>(4);
            for (int i = 0; i < 16; i++) w.AsSpan()[i] = 0.5f;
            for (int i = 0; i < 4; i++) b.AsSpan()[i] = 0.1f;

            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;

            graph.GetOrCreateTensor("X", new int[] { 1, 4 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 4 });

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
            using var batcher = new DynamicBatcher(session, maxBatchSize: 8, timeoutMicroseconds: 500);

            var tasks = new List<Task<NDArray<float>>>();
            for (int i = 0; i < 16; i++)
            {
                var input = new NDArray<float>(1, 4);
                for (int j = 0; j < 4; j++) input.AsSpan()[j] = 1.0f;
                tasks.Add(batcher.EnqueueAsync(input));
            }

            var results = await Task.WhenAll(tasks);

            Assert.Equal(16, results.Length);
            foreach (var res in results)
            {
                Assert.Equal(4, res.TotalLength);
                for (int j = 0; j < 4; j++)
                {
                    // Expected: 4 * 0.5 + 0.1 = 2.1
                    Assert.Equal(2.1f, res.AsSpan()[j], precision: 4);
                }
            }
        }
    }
}
