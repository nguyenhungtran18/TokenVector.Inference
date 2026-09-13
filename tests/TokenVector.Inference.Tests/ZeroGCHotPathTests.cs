using System;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class ZeroGCHotPathTests
    {
        [Fact]
        public void InferenceHotPath_AllocatesZeroBytesOnManagedHeap()
        {
            // Build a representative Graph: Input -> FusedLinear (Linear + Bias + GELU) -> Output
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");

            int inDim = 64;
            int outDim = 128;

            var w = new NDArray<float>(outDim, inDim);
            var b = new NDArray<float>(outDim);
            for (int i = 0; i < outDim * inDim; i++) w.AsSpan()[i] = 0.01f;
            for (int i = 0; i < outDim; i++) b.AsSpan()[i] = 0.05f;

            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;

            graph.GetOrCreateTensor("X", new int[] { 1, inDim });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, outDim });

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

            // Pre-allocate I/O buffers outside hot-path
            var inputTensor = new NDArray<float>(1, inDim);
            for (int i = 0; i < inDim; i++) inputTensor.AsSpan()[i] = 1.0f;

            var outputTensor = new NDArray<float>(1, outDim);

            NamedNDArray[] inArray = new NamedNDArray[1] { new("X", inputTensor) };
            NamedNDArray[] outArray = new NamedNDArray[1] { new("Y", outputTensor) };

            ReadOnlySpan<NamedNDArray> inSpan = inArray.AsSpan();
            Span<NamedNDArray> outSpan = outArray.AsSpan();

            // Warm-up to JIT compile any remaining methods
            for (int i = 0; i < 100; i++)
            {
                session.Run(inSpan, outSpan);
            }

            // Benchmark hot-path allocations
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();

            const int iterations = 1000;
            for (int i = 0; i < iterations; i++)
            {
                session.Run(inSpan, outSpan);
            }

            long afterAlloc = GC.GetAllocatedBytesForCurrentThread();
            long totalAllocated = afterAlloc - beforeAlloc;

            Assert.Equal(0, totalAllocated);
        }
    }
}
