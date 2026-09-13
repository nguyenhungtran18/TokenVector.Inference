using System;
using System.Collections.Generic;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.ONNX
{
    /// <summary>
    /// Translates ONNX Model Proto into TokenVector ExecutionGraph with Topological Sort (Kahn Algorithm).
    /// </summary>
    public static class GraphTranslator
    {
        public static ExecutionGraph Translate(OnnxModelProto modelProto)
        {
            var graphProto = modelProto.Graph;
            var execGraph = new ExecutionGraph
            {
                Name = string.IsNullOrEmpty(graphProto.Name) ? "OnnxExecutionGraph" : graphProto.Name
            };

            // 1. Extract Initializers
            foreach (var init in graphProto.Initializers)
            {
                int[] shape = new int[init.Dims.Count];
                for (int d = 0; d < init.Dims.Count; d++)
                    shape[d] = (int)init.Dims[d];

                float[] floatData = init.ExtractFloatArray();
                var ndArray = new NDArray<float>(shape.Length == 0 ? new int[] { 1 } : shape);
                if (floatData.Length > 0)
                {
                    floatData.AsSpan().CopyTo(ndArray.AsSpan());
                }

                execGraph.Initializers[init.Name] = ndArray;
                execGraph.GetOrCreateTensor(init.Name, ndArray.Shape, init.DataType);
            }

            // 2. Extract Graph Inputs
            foreach (var inp in graphProto.Inputs)
            {
                if (!execGraph.Initializers.ContainsKey(inp.Name))
                {
                    int[] shape = new int[inp.Dims.Count];
                    for (int d = 0; d < inp.Dims.Count; d++)
                        shape[d] = (int)(inp.Dims[d] > 0 ? inp.Dims[d] : 1);

                    execGraph.Inputs.Add(inp.Name);
                    execGraph.GetOrCreateTensor(inp.Name, shape, inp.DataType);
                }
            }

            // 3. Extract Graph Outputs
            foreach (var outp in graphProto.Outputs)
            {
                int[] shape = new int[outp.Dims.Count];
                for (int d = 0; d < outp.Dims.Count; d++)
                    shape[d] = (int)(outp.Dims[d] > 0 ? outp.Dims[d] : 1);

                execGraph.Outputs.Add(outp.Name);
                execGraph.GetOrCreateTensor(outp.Name, shape, outp.DataType);
            }

            // 4. Translate Nodes
            var rawNodes = new List<ExecutionNode>();
            for (int i = 0; i < graphProto.Nodes.Count; i++)
            {
                var onnxNode = graphProto.Nodes[i];
                var execNode = new ExecutionNode
                {
                    Id = i,
                    Name = string.IsNullOrEmpty(onnxNode.Name) ? $"{onnxNode.OpType}_{i}" : onnxNode.Name,
                    OpType = MapOpType(onnxNode.OpType)
                };

                foreach (var inName in onnxNode.Inputs)
                {
                    if (!string.IsNullOrEmpty(inName))
                    {
                        execNode.Inputs.Add(inName);
                        execGraph.GetOrCreateTensor(inName, Array.Empty<int>());
                    }
                }

                foreach (var outName in onnxNode.Outputs)
                {
                    if (!string.IsNullOrEmpty(outName))
                    {
                        execNode.Outputs.Add(outName);
                        execGraph.GetOrCreateTensor(outName, Array.Empty<int>());
                    }
                }

                // Parse Attributes
                if (onnxNode.Attributes.TryGetValue("alpha", out var attrAlpha))
                    execNode.Alpha = attrAlpha.F;
                if (onnxNode.Attributes.TryGetValue("beta", out var attrBeta))
                    execNode.Beta = attrBeta.F;
                if (onnxNode.Attributes.TryGetValue("transA", out var attrTransA))
                    execNode.TransA = (int)attrTransA.I;
                if (onnxNode.Attributes.TryGetValue("transB", out var attrTransB))
                    execNode.TransB = (int)attrTransB.I;
                if (onnxNode.Attributes.TryGetValue("epsilon", out var attrEps))
                    execNode.Epsilon = attrEps.F;
                if (onnxNode.Attributes.TryGetValue("axis", out var attrAxis))
                    execNode.Axis = (int)attrAxis.I;

                if (onnxNode.Attributes.TryGetValue("strides", out var attrStrides) && attrStrides.Ints.Count > 0)
                {
                    execNode.Strides = new int[attrStrides.Ints.Count];
                    for (int s = 0; s < attrStrides.Ints.Count; s++)
                        execNode.Strides[s] = (int)attrStrides.Ints[s];
                }
                if (onnxNode.Attributes.TryGetValue("pads", out var attrPads) && attrPads.Ints.Count > 0)
                {
                    execNode.Pads = new int[attrPads.Ints.Count];
                    for (int p = 0; p < attrPads.Ints.Count; p++)
                        execNode.Pads[p] = (int)attrPads.Ints[p];
                }

                // If Gemm has bias, treat as FusedLinear
                if (onnxNode.OpType == "Gemm" && execNode.Inputs.Count >= 3)
                {
                    execNode.OpType = OperatorType.FusedLinear;
                }

                rawNodes.Add(execNode);
            }

            // 5. Perform Topological Sort (Kahn's Algorithm)
            var sortedNodes = TopologicalSort(rawNodes, execGraph.Inputs, execGraph.Initializers.Keys);
            execGraph.Nodes.AddRange(sortedNodes);

            return execGraph;
        }

        private static OperatorType MapOpType(string opType)
        {
            return opType switch
            {
                "Gemm" => OperatorType.FusedLinear,
                "MatMul" => OperatorType.MatMul,
                "Conv" => OperatorType.Conv2D,
                "Relu" => OperatorType.Relu,
                "Gelu" => OperatorType.Gelu,
                "Sigmoid" => OperatorType.Sigmoid,
                "Tanh" => OperatorType.Tanh,
                "Add" => OperatorType.Add,
                "Sub" => OperatorType.Sub,
                "Mul" => OperatorType.Mul,
                "Div" => OperatorType.Div,
                "Softmax" => OperatorType.Softmax,
                "RMSNorm" or "SimplifiedLayerNormalization" => OperatorType.RMSNorm,
                "LayerNormalization" => OperatorType.LayerNorm,
                "Attention" or "MultiHeadAttention" => OperatorType.Attention,
                "Reshape" => OperatorType.Reshape,
                "Transpose" => OperatorType.Transpose,
                "Concat" => OperatorType.Concat,
                "QLinearMatMul" or "QuantizeLinear" or "DequantizeLinear" => OperatorType.QuantizedGemm,
                _ => OperatorType.Identity
            };
        }

        private static List<ExecutionNode> TopologicalSort(
            List<ExecutionNode> nodes,
            IEnumerable<string> graphInputs,
            IEnumerable<string> initializers)
        {
            var readyTensors = new HashSet<string>(graphInputs, StringComparer.Ordinal);
            foreach (var init in initializers)
                readyTensors.Add(init);

            var remainingNodes = new List<ExecutionNode>(nodes);
            var sorted = new List<ExecutionNode>(nodes.Count);

            while (remainingNodes.Count > 0)
            {
                bool progress = false;
                for (int i = 0; i < remainingNodes.Count; i++)
                {
                    var node = remainingNodes[i];
                    bool allInputsReady = true;

                    foreach (var inp in node.Inputs)
                    {
                        if (!readyTensors.Contains(inp))
                        {
                            allInputsReady = false;
                            break;
                        }
                    }

                    if (allInputsReady)
                    {
                        sorted.Add(node);
                        foreach (var outp in node.Outputs)
                            readyTensors.Add(outp);

                        remainingNodes.RemoveAt(i);
                        progress = true;
                        break;
                    }
                }

                if (!progress)
                {
                    // Fallback append remaining if cycle or unresolved dependencies
                    sorted.AddRange(remainingNodes);
                    break;
                }
            }

            return sorted;
        }
    }
}
