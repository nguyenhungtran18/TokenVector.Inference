using System;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class GraphOptimizationTests
    {
        [Fact]
        public void ConstantFoldingPass_FoldsStaticTranspose()
        {
            var graph = new ExecutionGraph();
            var init = new NDArray<float>(2, 3);
            var span = init.AsSpan();
            for (int i = 0; i < 6; i++) span[i] = i + 1;

            graph.Initializers["W"] = init;
            graph.GetOrCreateTensor("W", init.Shape);
            graph.GetOrCreateTensor("W_trans", new int[] { 3, 2 });

            var node = new ExecutionNode
            {
                Id = 0,
                Name = "Transpose_0",
                OpType = OperatorType.Transpose
            };
            node.Inputs.Add("W");
            node.Outputs.Add("W_trans");
            graph.Nodes.Add(node);

            var pass = new ConstantFoldingPass();
            bool modified = pass.Run(graph);

            Assert.True(modified);
            Assert.Empty(graph.Nodes);
            Assert.True(graph.Initializers.ContainsKey("W_trans"));
            Assert.Equal(3, graph.Initializers["W_trans"].Shape[0]);
            Assert.Equal(2, graph.Initializers["W_trans"].Shape[1]);
        }

        [Fact]
        public void OperatorFusionPass_FusesMatMulBiasActIntoFusedLinear()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            graph.Initializers["W"] = new NDArray<float>(4, 4);
            graph.Initializers["B"] = new NDArray<float>(4);

            graph.GetOrCreateTensor("X", new int[] { 1, 4 });
            graph.GetOrCreateTensor("W", new int[] { 4, 4 });
            graph.GetOrCreateTensor("B", new int[] { 4 });
            graph.GetOrCreateTensor("Inter1", new int[] { 1, 4 });
            graph.GetOrCreateTensor("Inter2", new int[] { 1, 4 });
            graph.GetOrCreateTensor("Y", new int[] { 1, 4 });

            var node1 = new ExecutionNode { Id = 0, OpType = OperatorType.MatMul };
            node1.Inputs.Add("X");
            node1.Inputs.Add("W");
            node1.Outputs.Add("Inter1");

            var node2 = new ExecutionNode { Id = 1, OpType = OperatorType.Add };
            node2.Inputs.Add("Inter1");
            node2.Inputs.Add("B");
            node2.Outputs.Add("Inter2");

            var node3 = new ExecutionNode { Id = 2, OpType = OperatorType.Relu };
            node3.Inputs.Add("Inter2");
            node3.Outputs.Add("Y");

            graph.Nodes.Add(node1);
            graph.Nodes.Add(node2);
            graph.Nodes.Add(node3);

            var pass = new OperatorFusionPass();
            bool modified = pass.Run(graph);

            Assert.True(modified);
            Assert.Single(graph.Nodes);
            var fusedNode = graph.Nodes[0];
            Assert.Equal(OperatorType.FusedLinear, fusedNode.OpType);
            Assert.Equal(ActivationFunction.ReLU, fusedNode.Activation);
            Assert.Equal("Y", fusedNode.Outputs[0]);
        }

        [Fact]
        public void DeadCodeEliminationPass_PrunesUnreachableNodes()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");

            graph.GetOrCreateTensor("X", new int[] { 1, 4 });
            graph.GetOrCreateTensor("Y", new int[] { 1, 4 });
            graph.GetOrCreateTensor("DeadOut", new int[] { 1, 4 });

            var liveNode = new ExecutionNode { Id = 0, OpType = OperatorType.Relu };
            liveNode.Inputs.Add("X");
            liveNode.Outputs.Add("Y");

            var deadNode = new ExecutionNode { Id = 1, OpType = OperatorType.Gelu };
            deadNode.Inputs.Add("X");
            deadNode.Outputs.Add("DeadOut");

            graph.Nodes.Add(liveNode);
            graph.Nodes.Add(deadNode);

            var pass = new DeadCodeEliminationPass();
            bool modified = pass.Run(graph);

            Assert.True(modified);
            Assert.Single(graph.Nodes);
            Assert.Equal("Y", graph.Nodes[0].Outputs[0]);
        }
    }
}
