using System;
using System.Collections.Generic;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Optimization
{
    /// <summary>
    /// Evaluates static subgraphs (e.g. constant weights reshape, transpose, permute) at graph compile time.
    /// </summary>
    public sealed class ConstantFoldingPass : IGraphPass
    {
        public string Name => "ConstantFoldingPass";

        public bool Run(ExecutionGraph graph)
        {
            bool modified = false;
            var nodesToRemove = new List<ExecutionNode>();

            foreach (var node in graph.Nodes)
            {
                // Check if all inputs are constant initializers
                bool allInputsConst = true;
                foreach (var inp in node.Inputs)
                {
                    if (!graph.Initializers.ContainsKey(inp))
                    {
                        allInputsConst = false;
                        break;
                    }
                }

                if (allInputsConst && node.Inputs.Count > 0 && node.Outputs.Count > 0)
                {
                    // Compute static result
                    string outName = node.Outputs[0];
                    var inArr = graph.Initializers[node.Inputs[0]];

                    if (node.OpType == OperatorType.Transpose && inArr.Shape.Length == 2)
                    {
                        var folded = new NDArray<float>(new int[] { inArr.Shape[1], inArr.Shape[0] });
                        var srcSpan = inArr.AsSpan();
                        var dstSpan = folded.AsSpan();
                        for (int r = 0; r < inArr.Shape[0]; r++)
                        {
                            for (int c = 0; c < inArr.Shape[1]; c++)
                            {
                                dstSpan[c * inArr.Shape[0] + r] = srcSpan[r * inArr.Shape[1] + c];
                            }
                        }
                        graph.Initializers[outName] = folded;
                        nodesToRemove.Add(node);
                        modified = true;
                    }
                    else if (node.OpType == OperatorType.Reshape)
                    {
                        if (graph.Tensors.TryGetValue(outName, out var outDesc))
                        {
                            var folded = new NDArray<float>(outDesc.Shape);
                            var srcSpan = inArr.AsSpan();
                            var dstSpan = folded.AsSpan();
                            srcSpan.Slice(0, Math.Min(srcSpan.Length, dstSpan.Length)).CopyTo(dstSpan);
                            graph.Initializers[outName] = folded;
                            nodesToRemove.Add(node);
                            modified = true;
                        }
                    }
                }
            }

            foreach (var n in nodesToRemove)
            {
                graph.Nodes.Remove(n);
            }

            return modified;
        }
    }
}
