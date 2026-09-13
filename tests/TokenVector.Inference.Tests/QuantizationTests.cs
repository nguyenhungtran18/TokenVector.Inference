using System;
using TokenVector.Inference.Quantization;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class QuantizationTests
    {
        [Fact]
        public void CalibrateMinMax_Symmetric_CalculatesCorrectScale()
        {
            float[] data = new float[] { -12.7f, 0f, 6.35f, 12.7f };
            var param = QuantizationEngine.CalibrateMinMax(data, symmetric: true);

            Assert.Equal(0.1f, param.Scale, precision: 5);
            Assert.Equal(0, param.ZeroPoint);
        }

        [Fact]
        public void QuantizeAndDequantize_Int8_PreservesValuesWithLowError()
        {
            float[] original = new float[] { -1.5f, -0.5f, 0.0f, 0.5f, 1.5f };
            var param = QuantizationEngine.CalibrateMinMax(original, symmetric: true);

            sbyte[] quantized = new sbyte[original.Length];
            QuantizationEngine.QuantizeToInt8Symmetric(original, quantized, param.Scale);

            float[] dequantized = new float[original.Length];
            QuantizationEngine.DequantizeFromInt8Symmetric(quantized, dequantized, param.Scale);

            float maxDiff = 0f;
            for (int i = 0; i < original.Length; i++)
            {
                float diff = MathF.Abs(original[i] - dequantized[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            Assert.True(maxDiff <= param.Scale, $"Quantization error {maxDiff} exceeds scale {param.Scale}");
        }

        [Fact]
        public void FP8_E4M3_ConversionRoundtripAccuracy()
        {
            float[] testValues = new float[] { 0.0f, 0.5f, 1.0f, -1.0f, 2.5f, 16.0f };

            foreach (float val in testValues)
            {
                var fp8 = FP8E4M3.FromFloat(val);
                float recovered = fp8.ToFloat();
                float relError = MathF.Abs(val - recovered) / (MathF.Abs(val) + 1e-5f);
                Assert.True(relError < 0.15f, $"E4M3 error too large for {val}: got {recovered}");
            }
        }

        [Fact]
        public void FP8_E5M2_ConversionRoundtripAccuracy()
        {
            float[] testValues = new float[] { 0.0f, 0.25f, 1.0f, -2.0f, 64.0f, 512.0f };

            foreach (float val in testValues)
            {
                var fp8 = FP8E5M2.FromFloat(val);
                float recovered = fp8.ToFloat();
                float relError = MathF.Abs(val - recovered) / (MathF.Abs(val) + 1e-5f);
                Assert.True(relError < 0.25f, $"E5M2 error too large for {val}: got {recovered}");
            }
        }

        [Fact]
        public unsafe void QuantizedMatMulInt8_MatchesFloatReference()
        {
            int M = 2, K = 4, N = 2;
            sbyte[] A = new sbyte[] { 10, 20, 30, 40, -10, -20, -30, -40 };
            sbyte[] W = new sbyte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            float[] bias = new float[] { 0.5f, 1.0f };
            float[] Y = new float[M * N];

            float scaleA = 0.1f;
            float scaleW = 0.05f;

            fixed (sbyte* pA = A)
            fixed (sbyte* pW = W)
            fixed (float* pBias = bias)
            fixed (float* pY = Y)
            {
                QuantizedMatMulKernels.MatMulInt8Symmetric(pA, pW, pBias, pY, M, N, K, scaleA, scaleW);
            }

            // Expected row 0 col 0: (10*1 + 20*2 + 30*3 + 40*4) * 0.1 * 0.05 + 0.5 = 300 * 0.005 + 0.5 = 1.5 + 0.5 = 2.0
            Assert.Equal(2.0f, Y[0], precision: 4);
        }
    }
}
