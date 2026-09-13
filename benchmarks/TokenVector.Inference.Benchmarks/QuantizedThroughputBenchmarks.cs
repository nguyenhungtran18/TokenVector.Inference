using System;
using BenchmarkDotNet.Attributes;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Quantization;

namespace TokenVector.Inference.Benchmarks
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public class QuantizedThroughputBenchmarks
    {
        private const int M = 16;
        private const int K = 256;
        private const int N = 256;

        private float[] _xFloat = null!;
        private float[] _wFloat = null!;
        private float[] _yFloat = null!;

        private sbyte[] _xInt8 = null!;
        private sbyte[] _wInt8 = null!;
        private float[] _yInt8 = null!;

        [GlobalSetup]
        public void Setup()
        {
            _xFloat = new float[M * K];
            _wFloat = new float[N * K];
            _yFloat = new float[M * N];

            for (int i = 0; i < _xFloat.Length; i++) _xFloat[i] = 0.5f;
            for (int i = 0; i < _wFloat.Length; i++) _wFloat[i] = 0.1f;

            _xInt8 = new sbyte[M * K];
            _wInt8 = new sbyte[N * K];
            _yInt8 = new float[M * N];

            for (int i = 0; i < _xInt8.Length; i++) _xInt8[i] = 50;
            for (int i = 0; i < _wInt8.Length; i++) _wInt8[i] = 10;
        }

        [Benchmark(Baseline = true)]
        public unsafe void FP32_DenseMatMul()
        {
            fixed (float* pX = _xFloat)
            fixed (float* pW = _wFloat)
            fixed (float* pY = _yFloat)
            {
                FusedKernels.FusedMatMulBiasAct(pX, pW, null, pY, M, N, K, ActivationFunction.None, transB: true);
            }
        }

        [Benchmark]
        public unsafe void INT8_AVX2_QuantizedMatMul()
        {
            fixed (sbyte* pA = _xInt8)
            fixed (sbyte* pW = _wInt8)
            fixed (float* pY = _yInt8)
            {
                QuantizedMatMulKernels.MatMulInt8Symmetric(pA, pW, null, pY, M, N, K, 0.01f, 0.01f);
            }
        }
    }
}
