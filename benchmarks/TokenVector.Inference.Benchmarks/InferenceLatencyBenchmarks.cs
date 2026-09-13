using System;
using BenchmarkDotNet.Attributes;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Benchmarks
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public class InferenceLatencyBenchmarks
    {
        private InferenceSession _mlpSession = null!;
        private InferenceSession _attentionSession = null!;
        private NamedNDArray[] _mlpInputs = null!;
        private NamedNDArray[] _mlpOutputs = null!;
        private NamedNDArray[] _attnInputs = null!;
        private NamedNDArray[] _attnOutputs = null!;

        [GlobalSetup]
        public void Setup()
        {
            // Setup MLP
            var mlpGraph = new ExecutionGraph();
            mlpGraph.Inputs.Add("X");
            mlpGraph.Outputs.Add("Y");
            var w1 = new NDArray<float>(64, 32);
            var b1 = new NDArray<float>(64);
            var w2 = new NDArray<float>(10, 64);
            var b2 = new NDArray<float>(10);
            for (int i = 0; i < 64 * 32; i++) w1.AsSpan()[i] = 0.01f;
            for (int i = 0; i < 64; i++) b1.AsSpan()[i] = 0.05f;
            for (int i = 0; i < 10 * 64; i++) w2.AsSpan()[i] = 0.02f;
            for (int i = 0; i < 10; i++) b2.AsSpan()[i] = 0.1f;
            mlpGraph.Initializers["W1"] = w1;
            mlpGraph.Initializers["B1"] = b1;
            mlpGraph.Initializers["W2"] = w2;
            mlpGraph.Initializers["B2"] = b2;
            mlpGraph.GetOrCreateTensor("X", new int[] { 1, 32 });
            mlpGraph.GetOrCreateTensor("W1", w1.Shape);
            mlpGraph.GetOrCreateTensor("B1", b1.Shape);
            mlpGraph.GetOrCreateTensor("W2", w2.Shape);
            mlpGraph.GetOrCreateTensor("B2", b2.Shape);
            mlpGraph.GetOrCreateTensor("H1", new int[] { 1, 64 });
            mlpGraph.GetOrCreateTensor("Y", new int[] { 1, 10 });

            var n1 = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.ReLU };
            n1.Inputs.Add("X"); n1.Inputs.Add("W1"); n1.Inputs.Add("B1"); n1.Outputs.Add("H1");
            var n2 = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.None };
            n2.Inputs.Add("H1"); n2.Inputs.Add("W2"); n2.Inputs.Add("B2"); n2.Outputs.Add("Y");
            mlpGraph.Nodes.Add(n1);
            mlpGraph.Nodes.Add(n2);
            _mlpSession = new InferenceSession(mlpGraph);

            var inX = new NDArray<float>(1, 32);
            for (int i = 0; i < 32; i++) inX.AsSpan()[i] = 1.0f;
            var outY = new NDArray<float>(1, 10);
            _mlpInputs = new NamedNDArray[] { new("X", inX) };
            _mlpOutputs = new NamedNDArray[] { new("Y", outY) };

            // Setup Attention
            var attnGraph = new ExecutionGraph();
            attnGraph.Inputs.Add("Q");
            attnGraph.Outputs.Add("Out");
            attnGraph.GetOrCreateTensor("Q", new int[] { 4, 32 });
            attnGraph.GetOrCreateTensor("Out", new int[] { 4, 32 });
            var attnNode = new ExecutionNode { OpType = OperatorType.Attention };
            attnNode.Inputs.Add("Q"); attnNode.Inputs.Add("Q"); attnNode.Inputs.Add("Q");
            attnNode.Outputs.Add("Out");
            attnGraph.Nodes.Add(attnNode);
            _attentionSession = new InferenceSession(attnGraph);

            var q = new NDArray<float>(4, 32);
            for (int i = 0; i < 128; i++) q.AsSpan()[i] = 0.1f;
            var outAttn = new NDArray<float>(4, 32);
            _attnInputs = new NamedNDArray[] { new("Q", q) };
            _attnOutputs = new NamedNDArray[] { new("Out", outAttn) };
        }

        [Benchmark(Baseline = true)]
        public void TokenVector_MLP_Inference()
        {
            _mlpSession.Run(_mlpInputs.AsSpan(), _mlpOutputs.AsSpan());
        }

        [Benchmark]
        public void TokenVector_Attention_Inference()
        {
            _attentionSession.Run(_attnInputs.AsSpan(), _attnOutputs.AsSpan());
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _mlpSession.Dispose();
            _attentionSession.Dispose();
        }
    }
}
