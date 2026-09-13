using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TokenVector.Inference.Optimization
{
    public enum ActivationFunction
    {
        None = 0,
        ReLU = 1,
        GELU = 2,
        SiLU = 3,
        Sigmoid = 4,
        Tanh = 5
    }

    /// <summary>
    /// Highly-optimized hardware-accelerated SIMD Fused Computation Kernels (AVX2 / FMA).
    /// Eliminates intermediate memory traffic by fusing MatMul/Conv with Bias, Norm and Activation in L1/L2 Cache.
    /// </summary>
    public static unsafe class FusedKernels
    {
        private const float Sqrt2OverPi = 0.7978845608028654f;
        private const float GeluCoeff = 0.044715f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ApplyActivation(float x, ActivationFunction act)
        {
            switch (act)
            {
                case ActivationFunction.ReLU:
                    return x > 0f ? x : 0f;
                case ActivationFunction.GELU:
                    return 0.5f * x * (1.0f + MathF.Tanh(Sqrt2OverPi * (x + GeluCoeff * x * x * x)));
                case ActivationFunction.SiLU:
                    return x / (1.0f + MathF.Exp(-x));
                case ActivationFunction.Sigmoid:
                    return 1.0f / (1.0f + MathF.Exp(-x));
                case ActivationFunction.Tanh:
                    return MathF.Tanh(x);
                default:
                    return x;
            }
        }

        /// <summary>
        /// Fused Linear: Y = Activation(X * W^T + Bias) or Activation(X * W + Bias).
        /// Dimensions: X (M x K), W (N x K if transB else K x N), Bias (N), Y (M x N).
        /// </summary>
        public static void FusedMatMulBiasAct(
            float* pX,
            float* pW,
            float* pBias,
            float* pY,
            int M,
            int N,
            int K,
            ActivationFunction activation,
            bool transB = true)
        {
            bool hasAvx2 = Avx2.IsSupported && Fma.IsSupported;

            for (int m = 0; m < M; m++)
            {
                float* rowX = pX + m * K;
                float* rowY = pY + m * N;

                for (int n = 0; n < N; n++)
                {
                    float sum = pBias != null ? pBias[n] : 0f;

                    if (transB)
                    {
                        float* rowW = pW + n * K;
                        if (hasAvx2 && K >= 8)
                        {
                            Vector256<float> vSum = Vector256<float>.Zero;
                            int k = 0;
                            int kLimit = K - 7;

                            for (; k < kLimit; k += 8)
                            {
                                Vector256<float> vx = Avx.LoadVector256(rowX + k);
                                Vector256<float> vw = Avx.LoadVector256(rowW + k);
                                vSum = Fma.MultiplyAdd(vx, vw, vSum);
                            }

                            Vector128<float> vLow = vSum.GetLower();
                            Vector128<float> vHigh = vSum.GetUpper();
                            Vector128<float> v128 = Sse.Add(vLow, vHigh);
                            v128 = Sse3.IsSupported ? Sse3.HorizontalAdd(v128, v128) : Sse.Add(v128, Sse.Shuffle(v128, v128, 0b01001110));
                            v128 = Sse3.IsSupported ? Sse3.HorizontalAdd(v128, v128) : Sse.Add(v128, Sse.Shuffle(v128, v128, 0b10110001));
                            sum += v128.ToScalar();

                            for (; k < K; k++)
                            {
                                sum += rowX[k] * rowW[k];
                            }
                        }
                        else
                        {
                            for (int k = 0; k < K; k++)
                            {
                                sum += rowX[k] * rowW[k];
                            }
                        }
                    }
                    else
                    {
                        // Column access in W
                        for (int k = 0; k < K; k++)
                        {
                            sum += rowX[k] * pW[k * N + n];
                        }
                    }

                    rowY[n] = ApplyActivation(sum, activation);
                }
            }
        }

        /// <summary>
        /// Fused RMSNorm Kernel: Computes X / sqrt(mean(X^2) + eps) * Gamma in a single pass.
        /// </summary>
        public static void FusedRMSNorm(
            float* pX,
            float* pGamma,
            float* pY,
            int N,
            int D,
            float eps = 1e-5f)
        {
            bool hasAvx2 = Avx2.IsSupported && Fma.IsSupported;

            for (int i = 0; i < N; i++)
            {
                float* xRow = pX + i * D;
                float* yRow = pY + i * D;

                float sumSq = 0f;

                if (hasAvx2 && D >= 8)
                {
                    Vector256<float> vSumSq = Vector256<float>.Zero;
                    int j = 0;
                    int jLimit = D - 7;

                    for (; j < jLimit; j += 8)
                    {
                        Vector256<float> vx = Avx.LoadVector256(xRow + j);
                        vSumSq = Fma.MultiplyAdd(vx, vx, vSumSq);
                    }

                    Vector128<float> vLow = vSumSq.GetLower();
                    Vector128<float> vHigh = vSumSq.GetUpper();
                    Vector128<float> v128 = Sse.Add(vLow, vHigh);
                    v128 = Sse3.IsSupported ? Sse3.HorizontalAdd(v128, v128) : Sse.Add(v128, Sse.Shuffle(v128, v128, 0b01001110));
                    v128 = Sse3.IsSupported ? Sse3.HorizontalAdd(v128, v128) : Sse.Add(v128, Sse.Shuffle(v128, v128, 0b10110001));
                    sumSq += v128.ToScalar();

                    for (; j < D; j++)
                    {
                        sumSq += xRow[j] * xRow[j];
                    }
                }
                else
                {
                    for (int j = 0; j < D; j++)
                    {
                        sumSq += xRow[j] * xRow[j];
                    }
                }

                float rsqrt = 1.0f / MathF.Sqrt((sumSq / D) + eps);

                if (hasAvx2 && D >= 8)
                {
                    Vector256<float> vRsqrt = Vector256.Create(rsqrt);
                    int j = 0;
                    int jLimit = D - 7;

                    for (; j < jLimit; j += 8)
                    {
                        Vector256<float> vx = Avx.LoadVector256(xRow + j);
                        Vector256<float> vg = pGamma != null ? Avx.LoadVector256(pGamma + j) : Vector256.Create(1.0f);
                        Vector256<float> vy = Avx.Multiply(Avx.Multiply(vx, vRsqrt), vg);
                        Avx.Store(yRow + j, vy);
                    }

                    for (; j < D; j++)
                    {
                        float g = pGamma != null ? pGamma[j] : 1.0f;
                        yRow[j] = xRow[j] * rsqrt * g;
                    }
                }
                else
                {
                    for (int j = 0; j < D; j++)
                    {
                        float g = pGamma != null ? pGamma[j] : 1.0f;
                        yRow[j] = xRow[j] * rsqrt * g;
                    }
                }
            }
        }

        /// <summary>
        /// Fused LayerNorm Kernel: Y = (X - mean) / sqrt(var + eps) * Gamma + Beta.
        /// </summary>
        public static void FusedLayerNorm(
            float* pX,
            float* pGamma,
            float* pBeta,
            float* pY,
            int N,
            int D,
            float eps = 1e-5f)
        {
            for (int i = 0; i < N; i++)
            {
                float* xRow = pX + i * D;
                float* yRow = pY + i * D;

                float sum = 0f;
                for (int j = 0; j < D; j++)
                    sum += xRow[j];
                float mean = sum / D;

                float varSum = 0f;
                for (int j = 0; j < D; j++)
                {
                    float diff = xRow[j] - mean;
                    varSum += diff * diff;
                }
                float invStd = 1.0f / MathF.Sqrt((varSum / D) + eps);

                for (int j = 0; j < D; j++)
                {
                    float g = pGamma != null ? pGamma[j] : 1.0f;
                    float b = pBeta != null ? pBeta[j] : 0.0f;
                    yRow[j] = (xRow[j] - mean) * invStd * g + b;
                }
            }
        }

        /// <summary>
        /// Fused Conv2D + BatchNorm + ReLU.
        /// </summary>
        public static void FusedConv2DBatchNormReLU(
            float* pX,
            float* pW,
            float* pBias,
            float* pMean,
            float* pVar,
            float* pGamma,
            float* pBeta,
            float* pY,
            int batch,
            int inChannels,
            int inH,
            int inW,
            int outChannels,
            int kH,
            int kW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            float eps = 1e-5f)
        {
            int outH = (inH + 2 * padH - kH) / strideH + 1;
            int outW = (inW + 2 * padW - kW) / strideW + 1;

            Span<float> effScale = stackalloc float[outChannels];
            Span<float> effBias = stackalloc float[outChannels];

            for (int oc = 0; oc < outChannels; oc++)
            {
                float g = pGamma != null ? pGamma[oc] : 1f;
                float b = pBeta != null ? pBeta[oc] : 0f;
                float m = pMean != null ? pMean[oc] : 0f;
                float v = pVar != null ? pVar[oc] : 1f;
                float origBias = pBias != null ? pBias[oc] : 0f;

                float invStd = 1.0f / MathF.Sqrt(v + eps);
                float scale = g * invStd;
                effScale[oc] = scale;
                effBias[oc] = b + (origBias - m) * scale;
            }

            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    float* wChannel = pW + oc * inChannels * kH * kW;
                    float scale = effScale[oc];
                    float bias = effBias[oc];

                    for (int oh = 0; oh < outH; oh++)
                    {
                        int ihBase = oh * strideH - padH;
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int iwBase = ow * strideW - padW;
                            float sum = 0f;

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                float* xSlice = pX + ((b * inChannels + ic) * inH) * inW;
                                float* wSlice = wChannel + (ic * kH) * kW;

                                for (int kh = 0; kh < kH; kh++)
                                {
                                    int ih = ihBase + kh;
                                    if (ih >= 0 && ih < inH)
                                    {
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int iw = iwBase + kw;
                                            if (iw >= 0 && iw < inW)
                                            {
                                                sum += xSlice[ih * inW + iw] * wSlice[kh * kW + kw];
                                            }
                                        }
                                    }
                                }
                            }

                            float val = sum * scale + bias;
                            pY[((b * outChannels + oc) * outH + oh) * outW + ow] = val > 0f ? val : 0f;
                        }
                    }
                }
            }
        }
    }
}
