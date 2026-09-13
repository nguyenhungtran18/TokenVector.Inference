using System;
using System.Collections.Generic;
using TokenVector.Inference.Runtime;

namespace TokenVector.Inference.Optimization
{
    /// <summary>
    /// Eliminates unused operations and intermediate tensors that do not contribute to graph outputs.
    /// </summary>
    public sealed class DeadCodeEliminationPass : IGraphPass
    {
        public string Name => "DeadCodeEliminationPass";

        public bool Run(ExecutionGraph graph)
        {
            var requiredTensors = new HashSet<string>(graph.Outputs, StringComparer.Ordinal);
            var activeNodes = new HashSet<ExecutionNode>();

            // Traverse backward from outputs
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = graph.Nodes.Count - 1; i >= 0; i--)
                {
                    var node = graph.Nodes[i];
                    if (activeNodes.Contains(node)) continue;

                    bool contributes = false;
                    foreach (var outp in node.Outputs)
                    {
                        if (requiredTensors.Contains(outp))
                        {
                            contributes = true;
                            break;
                        }
                    }

                    if (contributes)
                    {
                        activeNodes.Add(node);
                        foreach (var inp in node.Inputs)
                        {
                            if (requiredTensors.Add(inp))
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }

            int originalCount = graph.Nodes.Count;
            graph.Nodes.RemoveAll(n => !activeNodes.Contains(n));

            // Clean unused tensors
            var unusedTensors = new List<string>();
            foreach (var tName in graph.Tensors.Keys)
            {
                if (!requiredTensors.Contains(tName) && !graph.Inputs.Contains(tName) && !graph.Outputs.Contains(tName))
                {
                    unusedTensors.Add(tName);
                }
            }

            foreach (var tName in unusedTensors)
            {
                graph.Tensors.Remove(tName);
                graph.Initializers.Remove(tName);
            }

            return graph.Nodes.Count < originalCount;
        }
    }
}
