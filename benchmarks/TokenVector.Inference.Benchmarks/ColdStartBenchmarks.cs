using System;
using BenchmarkDotNet.Attributes;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Benchmarks
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public class ColdStartBenchmarks
    {
        private ExecutionGraph _graph = null!;

        [GlobalSetup]
        public void Setup()
        {
            _graph = new ExecutionGraph();
            _graph.Inputs.Add("X");
            _graph.Outputs.Add("Y");

            var w1 = new NDArray<float>(64, 32);
            var b1 = new NDArray<float>(64);
            var w2 = new NDArray<float>(10, 64);
            var b2 = new NDArray<float>(10);

            _graph.Initializers["W1"] = w1;
            _graph.Initializers["B1"] = b1;
            _graph.Initializers["W2"] = w2;
            _graph.Initializers["B2"] = b2;

            _graph.GetOrCreateTensor("X", new int[] { 1, 32 });
            _graph.GetOrCreateTensor("W1", w1.Shape);
            _graph.GetOrCreateTensor("B1", b1.Shape);
            _graph.GetOrCreateTensor("W2", w2.Shape);
            _graph.GetOrCreateTensor("B2", b2.Shape);
            _graph.GetOrCreateTensor("H1", new int[] { 1, 64 });
            _graph.GetOrCreateTensor("Y", new int[] { 1, 10 });

            var n1 = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.ReLU };
            n1.Inputs.Add("X"); n1.Inputs.Add("W1"); n1.Inputs.Add("B1");
            n1.Outputs.Add("H1");

            var n2 = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.None };
            n2.Inputs.Add("H1"); n2.Inputs.Add("W2"); n2.Inputs.Add("B2");
            n2.Outputs.Add("Y");

            _graph.Nodes.Add(n1);
            _graph.Nodes.Add(n2);
        }

        [Benchmark(Baseline = true)]
        public void TokenVector_ColdStart_SessionInit()
        {
            using var session = new InferenceSession(_graph, applyOptimizations: true);
        }
    }
}
