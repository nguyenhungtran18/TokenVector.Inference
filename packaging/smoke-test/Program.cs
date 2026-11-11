using System;
using System.Collections.Generic;

// Smoke test: xác nhận TokenVector.Inference.dll gọi được từ .NET.
// Chạy qua run.ps1 (tự đảm bảo DLL đã build + copy runtime DLLs).
//
// API (mức IL, không namespace — mọi class/method global):
//   new ExecutionGraph(1, name, 0)   -- tạo graph
//   g.inputs.Add / outputs.Add       -- List<string>
//   g.initializers[...] = list[f32]  -- Dictionary<string, List<float>>
//   TKVApp._nd_f32(n)                -- List<float> (mảng khởi tạo)
//   TKVApp.new_node("gemm", 2)       -- ExecutionNode (2 = OperatorType_FusedLinear)
//   TKVApp.session_from_graph(g, 1)  -- InferenceSession (1 = bật optimization passes)
//   session.run_single(x, "X", "Y")  -- forward 1 sample
static class Smoke
{
    static int Main()
    {
        try
        {
            var g = new ExecutionGraph(1, "smoke", 0);
            g.inputs.Add("X");
            g.outputs.Add("Y");

            var w = TKVApp._nd_f32(32);
            for (int i = 0; i < 32; i++) w[i] = 0.5f;
            var b = TKVApp._nd_f32(8);
            for (int i = 0; i < 8; i++) b[i] = 0.25f;
            g.initializers["W"] = w;
            g.initializers["B"] = b;

            g.get_or_create_tensor("X", new List<long> { 1, 4 }, 0);
            g.get_or_create_tensor("W", new List<long> { 8, 4 }, 0);
            g.get_or_create_tensor("B", new List<long> { 8 }, 0);
            g.get_or_create_tensor("Y", new List<long> { 1, 8 }, 0);

            var node = TKVApp.new_node("gemm", 2);   // OperatorType_FusedLinear
            node.activation = 1;                     // ActivationFunction_Relu
            node.inputs.Add("X"); node.inputs.Add("W"); node.inputs.Add("B");
            node.outputs.Add("Y");
            g.nodes.Add(node);

            var session = TKVApp.session_from_graph(g, 1);

            var x = TKVApp._nd_f32(4);
            for (int i = 0; i < 4; i++) x[i] = 1.0f;
            var y = session.run_single(x, "X", "Y");
            Console.WriteLine("Y[0] = " + y[0] + "  (expect 2.25)");
            session.close();
            Console.WriteLine("SMOKE OK");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine("SMOKE FAIL: " + ex.GetType().Name + ": " + ex.Message);
            return 1;
        }
    }
}
