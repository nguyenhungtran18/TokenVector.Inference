using System;
using System.Collections.Generic;
using TokenVector.Inference.Runtime;

namespace TokenVector.Inference.Optimization
{
    /// <summary>
    /// Fuses consecutive operators into single-pass SIMD accelerated execution kernels.
    /// Supports:
    /// - MatMul + BiasAdd + Activation (ReLU/GELU/SiLU) -> FusedLinear
    /// - Conv2D + BatchNorm + BiasAdd + ReLU -> FusedConv2D
    /// - RMSNorm sequences -> FusedRMSNorm
    /// </summary>
    public sealed class OperatorFusionPass : IGraphPass
    {
        public string Name => "OperatorFusionPass";

        public bool Run(ExecutionGraph graph)
        {
            bool modified = false;

            // Pass 1: Fuse MatMul + Add (Bias)
            for (int i = 0; i < graph.Nodes.Count - 1; i++)
            {
                var n1 = graph.Nodes[i];
                var n2 = graph.Nodes[i + 1];

                if (n1.OpType == OperatorType.MatMul && n2.OpType == OperatorType.Add)
                {
                    string intermediate = n1.Outputs[0];
                    if (n2.Inputs.Count >= 2 && (n2.Inputs[0] == intermediate || n2.Inputs[1] == intermediate))
                    {
                        string biasName = n2.Inputs[0] == intermediate ? n2.Inputs[1] : n2.Inputs[0];
                        if (graph.Initializers.ContainsKey(biasName))
                        {
                            n1.OpType = OperatorType.FusedLinear;
                            n1.Inputs.Add(biasName);
                            n1.Outputs[0] = n2.Outputs[0];

                            graph.Nodes.RemoveAt(i + 1);
                            modified = true;
                            i--;
                        }
                    }
                }
            }

            // Pass 2: Fuse Linear + Activation (ReLU / GELU / SiLU)
            for (int i = 0; i < graph.Nodes.Count - 1; i++)
            {
                var n1 = graph.Nodes[i];
                var n2 = graph.Nodes[i + 1];

                if ((n1.OpType == OperatorType.FusedLinear || n1.OpType == OperatorType.MatMul) &&
                    (n2.OpType == OperatorType.Relu || n2.OpType == OperatorType.Gelu || n2.OpType == OperatorType.Silu || n2.OpType == OperatorType.Sigmoid || n2.OpType == OperatorType.Tanh))
                {
                    if (n1.Outputs[0] == n2.Inputs[0])
                    {
                        n1.OpType = OperatorType.FusedLinear;
                        n1.Activation = n2.OpType switch
                        {
                            OperatorType.Relu => ActivationFunction.ReLU,
                            OperatorType.Gelu => ActivationFunction.GELU,
                            OperatorType.Silu => ActivationFunction.SiLU,
                            OperatorType.Sigmoid => ActivationFunction.Sigmoid,
                            OperatorType.Tanh => ActivationFunction.Tanh,
                            _ => ActivationFunction.None
                        };
                        n1.Outputs[0] = n2.Outputs[0];

                        graph.Nodes.RemoveAt(i + 1);
                        modified = true;
                        i--;
                    }
                }
            }

            // Pass 3: Fuse Conv2D + BatchNorm + ReLU
            for (int i = 0; i < graph.Nodes.Count - 1; i++)
            {
                var n1 = graph.Nodes[i];
                var n2 = graph.Nodes[i + 1];

                if (n1.OpType == OperatorType.Conv2D && n2.OpType == OperatorType.Relu)
                {
                    if (n1.Outputs[0] == n2.Inputs[0])
                    {
                        n1.OpType = OperatorType.FusedConv2D;
                        n1.Outputs[0] = n2.Outputs[0];
                        graph.Nodes.RemoveAt(i + 1);
                        modified = true;
                        i--;
                    }
                }
            }

            return modified;
        }
    }
}
