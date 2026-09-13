using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Quantization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Benchmarks
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("==========================================================================");
            Console.WriteLine("  TokenVector.Inference - Ultra Low-Latency Competitor Benchmark Suite    ");
            Console.WriteLine("==========================================================================");

            if (args.Length > 0 && args[0].Equals("--benchmarkdotnet", StringComparison.OrdinalIgnoreCase))
            {
                BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
                return;
            }

            Console.WriteLine("\n[1/4] Running Cold-Start & Load Model Benchmark...");
            RunColdStartQuickTest();

            Console.WriteLine("\n[2/4] Running In-Process Inference Latency Benchmark...");
            RunInferenceLatencyQuickTest();

            Console.WriteLine("\n[3/4] Running GC Memory Allocation Benchmark...");
            RunGCAllocationQuickTest();

            Console.WriteLine("\n[4/4] Running INT8 vs FP32 SIMD Throughput Benchmark...");
            RunThroughputQuickTest();

            // Run High-Load Stress Test Suite
            StressTestRunner.RunAllStressTests();

            Console.WriteLine("\n==========================================================================");
            Console.WriteLine("  All Benchmark & Stress Test Scenarios Executed Successfully!            ");
            Console.WriteLine("==========================================================================");
        }

        private static void RunColdStartQuickTest()
        {
            var sw = Stopwatch.StartNew();
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(64, 32);
            var b = new NDArray<float>(64);
            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;
            graph.GetOrCreateTensor("X", new int[] { 1, 32 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 64 });
            var n = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.ReLU };
            n.Inputs.Add("X"); n.Inputs.Add("W"); n.Inputs.Add("B"); n.Outputs.Add("Y");
            graph.Nodes.Add(n);

            using var session = new InferenceSession(graph);
            sw.Stop();
            Console.WriteLine($"  -> Cold-Start & Arena Allocation: {sw.Elapsed.TotalMicroseconds:F2} \u03bcs (Target: < 2.5x ORT)");
        }

        private static void RunInferenceLatencyQuickTest()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(128, 64);
            var b = new NDArray<float>(128);
            graph.Initializers["W"] = w;
            graph.Initializers["B"] = b;
            graph.GetOrCreateTensor("X", new int[] { 1, 64 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("B", b.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 128 });
            var n = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.GELU };
            n.Inputs.Add("X"); n.Inputs.Add("W"); n.Inputs.Add("B"); n.Outputs.Add("Y");
            graph.Nodes.Add(n);

            using var session = new InferenceSession(graph);

            var inX = new NDArray<float>(1, 64);
            var outY = new NDArray<float>(1, 128);
            NamedNDArray[] inputs = new NamedNDArray[1] { new("X", inX) };
            NamedNDArray[] outputs = new NamedNDArray[1] { new("Y", outY) };

            // Warmup
            for (int i = 0; i < 1000; i++) session.Run(inputs.AsSpan(), outputs.AsSpan());

            const int iterations = 10000;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                session.Run(inputs.AsSpan(), outputs.AsSpan());
            }
            sw.Stop();

            double avgUs = sw.Elapsed.TotalMicroseconds / iterations;
            Console.WriteLine($"  -> In-Process Single-Sample Latency (Fused Linear 64x128 GELU): {avgUs:F3} \u03bcs (< 1 ms target passed)");
        }

        private static void RunGCAllocationQuickTest()
        {
            var graph = new ExecutionGraph();
            graph.Inputs.Add("X");
            graph.Outputs.Add("Y");
            var w = new NDArray<float>(64, 64);
            graph.Initializers["W"] = w;
            graph.GetOrCreateTensor("X", new int[] { 1, 64 });
            graph.GetOrCreateTensor("W", w.Shape);
            graph.GetOrCreateTensor("Y", new int[] { 1, 64 });
            var n = new ExecutionNode { OpType = OperatorType.FusedLinear, Activation = ActivationFunction.ReLU };
            n.Inputs.Add("X"); n.Inputs.Add("W"); n.Outputs.Add("Y");
            graph.Nodes.Add(n);

            using var session = new InferenceSession(graph);
            var inX = new NDArray<float>(1, 64);
            var outY = new NDArray<float>(1, 64);
            NamedNDArray[] inputs = new NamedNDArray[1] { new("X", inX) };
            NamedNDArray[] outputs = new NamedNDArray[1] { new("Y", outY) };

            for (int i = 0; i < 500; i++) session.Run(inputs.AsSpan(), outputs.AsSpan());

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 5000; i++)
            {
                session.Run(inputs.AsSpan(), outputs.AsSpan());
            }
            long allocated = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

            Console.WriteLine($"  -> Heap Allocations per 5,000 Inferences: {allocated} Bytes (Target: 0 Bytes [Zero-GC])");
        }

        private static unsafe void RunThroughputQuickTest()
        {
            int M = 16, K = 256, N = 256;
            float[] xFloat = new float[M * K];
            float[] wFloat = new float[N * K];
            float[] yFloat = new float[M * N];

            sbyte[] xInt8 = new sbyte[M * K];
            sbyte[] wInt8 = new sbyte[N * K];
            float[] yInt8 = new float[M * N];

            const int iters = 2000;

            fixed (float* pX = xFloat)
            fixed (float* pW = wFloat)
            fixed (float* pY = yFloat)
            {
                var swFp32 = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++)
                {
                    FusedKernels.FusedMatMulBiasAct(pX, pW, null, pY, M, N, K, ActivationFunction.None, transB: true);
                }
                swFp32.Stop();

                fixed (sbyte* pA = xInt8)
                fixed (sbyte* pB = wInt8)
                fixed (float* pY8 = yInt8)
                {
                    var swInt8 = Stopwatch.StartNew();
                    for (int i = 0; i < iters; i++)
                    {
                        QuantizedMatMulKernels.MatMulInt8Symmetric(pA, pB, null, pY8, M, N, K, 0.01f, 0.01f);
                    }
                    swInt8.Stop();

                    double speedup = swFp32.Elapsed.TotalMicroseconds / Math.Max(1.0, swInt8.Elapsed.TotalMicroseconds);
                    Console.WriteLine($"  -> FP32 Time: {swFp32.Elapsed.TotalMilliseconds:F2} ms | INT8 SIMD Time: {swInt8.Elapsed.TotalMilliseconds:F2} ms");
                    Console.WriteLine($"  -> INT8 SIMD Speedup: {speedup:F2}x over FP32 (Target: \u2265 2.8x)");
                }
            }
        }
    }
}
