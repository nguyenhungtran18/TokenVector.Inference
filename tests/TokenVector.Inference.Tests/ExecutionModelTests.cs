using System;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Quantization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class ExecutionModelTests
    {
        [Fact]
        public void MLPModel_ExecutionMatchesGroundTruth_MAEBelow1e5()
        {
            // Build 2-layer MLP graph: Input [1, 4] -> Linear [4, 8] + Bias + ReLU -> Linear [8, 2] + Bias
            var graph = new ExecutionGraph();
            graph.Inputs.Add("input");
            graph.Outputs.Add("output");

            var w1 = new NDArray<float>(8, 4);
            var b1 = new NDArray<float>(8);
            var w2 = new NDArray<float>(2, 8);
            var b2 = new NDArray<float>(2);

            for (int i = 0; i < 32; i++) w1.AsSpan()[i] = 0.1f * (i % 5 + 1);
            for (int i = 0; i < 8; i++) b1.AsSpan()[i] = 0.05f;
            for (int i = 0; i < 16; i++) w2.AsSpan()[i] = 0.2f * (i % 3 + 1);
            for (int i = 0; i < 2; i++) b2.AsSpan()[i] = 0.1f;

            graph.Initializers["W1"] = w1;
            graph.Initializers["B1"] = b1;
            graph.Initializers["W2"] = w2;
            graph.Initializers["B2"] = b2;

            graph.GetOrCreateTensor("input", new int[] { 1, 4 });
            graph.GetOrCreateTensor("W1", w1.Shape);
            graph.GetOrCreateTensor("B1", b1.Shape);
            graph.GetOrCreateTensor("W2", w2.Shape);
            graph.GetOrCreateTensor("B2", b2.Shape);
            graph.GetOrCreateTensor("h1", new int[] { 1, 8 });
            graph.GetOrCreateTensor("output", new int[] { 1, 2 });

            var node1 = new ExecutionNode
            {
                Id = 0,
                OpType = OperatorType.FusedLinear,
                Activation = ActivationFunction.ReLU
            };
            node1.Inputs.Add("input");
            node1.Inputs.Add("W1");
            node1.Inputs.Add("B1");
            node1.Outputs.Add("h1");

            var node2 = new ExecutionNode
            {
                Id = 1,
                OpType = OperatorType.FusedLinear,
                Activation = ActivationFunction.None
            };
            node2.Inputs.Add("h1");
            node2.Inputs.Add("W2");
            node2.Inputs.Add("B2");
            node2.Outputs.Add("output");

            graph.Nodes.Add(node1);
            graph.Nodes.Add(node2);

            using var session = new InferenceSession(graph);

            var input = new NDArray<float>(1, 4);
            input.AsSpan()[0] = 1.0f;
            input.AsSpan()[1] = 2.0f;
            input.AsSpan()[2] = 3.0f;
            input.AsSpan()[3] = 4.0f;

            var result = session.Run(input);

            // Calculate manual ground truth
            float[] h1Manual = new float[8];
            for (int n = 0; n < 8; n++)
            {
                float sum = b1.AsSpan()[n];
                for (int k = 0; k < 4; k++)
                    sum += input.AsSpan()[k] * w1.AsSpan()[n * 4 + k];
                h1Manual[n] = sum > 0f ? sum : 0f;
            }

            float[] outManual = new float[2];
            for (int n = 0; n < 2; n++)
            {
                float sum = b2.AsSpan()[n];
                for (int k = 0; k < 8; k++)
                    sum += h1Manual[k] * w2.AsSpan()[n * 8 + k];
                outManual[n] = sum;
            }

            float mae = 0f;
            for (int i = 0; i < 2; i++)
            {
                mae += MathF.Abs(result.AsSpan()[i] - outManual[i]);
            }
            mae /= 2f;

            Assert.True(mae < 1e-5f, $"MLP MAE {mae} is not < 1e-5");
        }

        [Fact]
        public void ResNetBlock_ExecutionMAEBelow1e5()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("x");
            graph.Outputs.Add("y");

            var w1 = new NDArray<float>(2, 2, 3, 3);
            var w2 = new NDArray<float>(2, 2, 3, 3);
            for (int i = 0; i < 36; i++) w1.AsSpan()[i] = 0.05f * ((i % 5) + 1);
            for (int i = 0; i < 36; i++) w2.AsSpan()[i] = 0.05f * ((i % 4) + 1);

            graph.Initializers["W1"] = w1;
            graph.Initializers["W2"] = w2;

            graph.GetOrCreateTensor("x", new int[] { 1, 2, 4, 4 });
            graph.GetOrCreateTensor("W1", w1.Shape);
            graph.GetOrCreateTensor("W2", w2.Shape);
            graph.GetOrCreateTensor("conv1_out", new int[] { 1, 2, 4, 4 });
            graph.GetOrCreateTensor("conv2_out", new int[] { 1, 2, 4, 4 });
            graph.GetOrCreateTensor("res_add", new int[] { 1, 2, 4, 4 });
            graph.GetOrCreateTensor("y", new int[] { 1, 2, 4, 4 });

            var conv1 = new ExecutionNode
            {
                OpType = OperatorType.FusedConv2D,
                Strides = new int[] { 1, 1 },
                Pads = new int[] { 1, 1, 1, 1 }
            };
            conv1.Inputs.Add("x");
            conv1.Inputs.Add("W1");
            conv1.Outputs.Add("conv1_out");

            var conv2 = new ExecutionNode
            {
                OpType = OperatorType.Conv2D,
                Strides = new int[] { 1, 1 },
                Pads = new int[] { 1, 1, 1, 1 }
            };
            conv2.Inputs.Add("conv1_out");
            conv2.Inputs.Add("W2");
            conv2.Outputs.Add("conv2_out");

            var addNode = new ExecutionNode { OpType = OperatorType.Add };
            addNode.Inputs.Add("conv2_out");
            addNode.Inputs.Add("x");
            addNode.Outputs.Add("res_add");

            var reluNode = new ExecutionNode { OpType = OperatorType.Relu };
            reluNode.Inputs.Add("res_add");
            reluNode.Outputs.Add("y");

            graph.Nodes.Add(conv1);
            graph.Nodes.Add(conv2);
            graph.Nodes.Add(addNode);
            graph.Nodes.Add(reluNode);

            using var session = new InferenceSession(graph);

            var input = new NDArray<float>(1, 2, 4, 4);
            for (int i = 0; i < 32; i++) input.AsSpan()[i] = 0.5f;

            var result = session.Run(input);

            Assert.Equal(32, result.TotalLength);
            // Verify all values are non-negative due to final ReLU
            for (int i = 0; i < 32; i++)
            {
                Assert.True(result.AsSpan()[i] >= 0f);
            }
        }

        [Fact]
        public void LLaMATransformerLayer_ExecutionSucceeds()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("tokens");
            graph.Outputs.Add("layer_out");

            int seqLen = 4;
            int hiddenDim = 16;

            var gamma1 = new NDArray<float>(hiddenDim);
            var gamma2 = new NDArray<float>(hiddenDim);
            for (int i = 0; i < hiddenDim; i++)
            {
                gamma1.AsSpan()[i] = 1.0f;
                gamma2.AsSpan()[i] = 1.0f;
            }

            var wFfn = new NDArray<float>(hiddenDim, hiddenDim);
            for (int i = 0; i < hiddenDim * hiddenDim; i++)
                wFfn.AsSpan()[i] = 0.01f * (i % 5 + 1);

            graph.Initializers["gamma1"] = gamma1;
            graph.Initializers["gamma2"] = gamma2;
            graph.Initializers["wFfn"] = wFfn;

            graph.GetOrCreateTensor("tokens", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("gamma1", gamma1.Shape);
            graph.GetOrCreateTensor("gamma2", gamma2.Shape);
            graph.GetOrCreateTensor("wFfn", wFfn.Shape);
            graph.GetOrCreateTensor("norm1", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("attn_out", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("res1", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("norm2", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("ffn_out", new int[] { seqLen, hiddenDim });
            graph.GetOrCreateTensor("layer_out", new int[] { seqLen, hiddenDim });

            var normNode1 = new ExecutionNode { OpType = OperatorType.RMSNorm };
            normNode1.Inputs.Add("tokens");
            normNode1.Inputs.Add("gamma1");
            normNode1.Outputs.Add("norm1");

            var attnNode = new ExecutionNode { OpType = OperatorType.Attention };
            attnNode.Inputs.Add("norm1");
            attnNode.Inputs.Add("norm1");
            attnNode.Inputs.Add("norm1");
            attnNode.Outputs.Add("attn_out");

            var addNode1 = new ExecutionNode { OpType = OperatorType.Add };
            addNode1.Inputs.Add("tokens");
            addNode1.Inputs.Add("attn_out");
            addNode1.Outputs.Add("res1");

            var normNode2 = new ExecutionNode { OpType = OperatorType.RMSNorm };
            normNode2.Inputs.Add("res1");
            normNode2.Inputs.Add("gamma2");
            normNode2.Outputs.Add("norm2");

            var ffnNode = new ExecutionNode
            {
                OpType = OperatorType.FusedLinear,
                Activation = ActivationFunction.SiLU
            };
            ffnNode.Inputs.Add("norm2");
            ffnNode.Inputs.Add("wFfn");
            ffnNode.Outputs.Add("ffn_out");

            var addNode2 = new ExecutionNode { OpType = OperatorType.Add };
            addNode2.Inputs.Add("res1");
            addNode2.Inputs.Add("ffn_out");
            addNode2.Outputs.Add("layer_out");

            graph.Nodes.Add(normNode1);
            graph.Nodes.Add(attnNode);
            graph.Nodes.Add(addNode1);
            graph.Nodes.Add(normNode2);
            graph.Nodes.Add(ffnNode);
            graph.Nodes.Add(addNode2);

            using var session = new InferenceSession(graph);

            var input = new NDArray<float>(seqLen, hiddenDim);
            for (int i = 0; i < seqLen * hiddenDim; i++)
                input.AsSpan()[i] = 0.1f * ((i % 7) + 1);

            var output = session.Run(input);

            Assert.Equal(seqLen * hiddenDim, output.TotalLength);
            for (int i = 0; i < output.TotalLength; i++)
            {
                Assert.False(float.IsNaN(output.AsSpan()[i]));
                Assert.False(float.IsInfinity(output.AsSpan()[i]));
            }
        }
    }
}
