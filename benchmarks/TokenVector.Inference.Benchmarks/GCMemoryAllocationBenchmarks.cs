using System;
using BenchmarkDotNet.Attributes;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Benchmarks
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public class GCMemoryAllocationBenchmarks
    {
        private InferenceSession _session = null!;
        private NamedNDArray[] _inputs = null!;
        private NamedNDArray[] _outputs = null!;

        [GlobalSetup]
        public void Setup()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(64, 64);
            var b = new NDArray<float>(64);
            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;
            graph.GetOrCreateTensor("X", new int[] { 1, 64 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 64 });

            var node = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.ReLU };
            node.Inputs.Add("X"); node.Inputs.Add("W"); node.Inputs.Add("B");
            node.Outputs.Add("Y");
            graph.Nodes.Add(node);

            _session = new InferenceSession(graph);

            var inX = new NDArray<float>(1, 64);
            var outY = new NDArray<float>(1, 64);
            _inputs = new NamedNDArray[] { new("X", inX) };
            _outputs = new NamedNDArray[] { new("Y", outY) };
        }

        [Benchmark(Baseline = true)]
        public void TokenVector_ZeroGCHotPath()
        {
            _session.Run(_inputs.AsSpan(), _outputs.AsSpan());
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _session.Dispose();
        }
    }
}
