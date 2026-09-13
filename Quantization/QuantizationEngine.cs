using System;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Quantization
{
    /// <summary>
    /// Engine for Static Post-Training Quantization (PTQ) and Dynamic Quantization.
    /// Supports MinMax, KL-Divergence calibration, and symmetric/asymmetric mapping.
    /// </summary>
    public static class QuantizationEngine
    {
        public static QuantizationParams CalibrateMinMax(ReadOnlySpan<float> data, bool symmetric = true)
        {
            if (data.Length == 0) return QuantizationParams.Identity;

            float min = float.MaxValue;
            float max = float.MinValue;

            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            if (symmetric)
            {
                float absMax = MathF.Max(MathF.Abs(min), MathF.Abs(max));
                if (absMax < 1e-8f) absMax = 1e-8f;
                float scale = absMax / 127.0f;
                return new QuantizationParams(scale, 0, -absMax, absMax);
            }
            else
            {
                if (MathF.Abs(max - min) < 1e-8f)
                    max = min + 1e-8f;

                float scale = (max - min) / 255.0f;
                int zeroPoint = (int)MathF.Round(-min / scale) - 128;
                zeroPoint = Math.Clamp(zeroPoint, -128, 127);
                return new QuantizationParams(scale, zeroPoint, min, max);
            }
        }

        public static QuantizationParams CalibrateKLDivergence(ReadOnlySpan<float> data, int numBins = 2048, int targetBins = 128)
        {
            if (data.Length == 0) return QuantizationParams.Identity;

            float absMax = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                float a = MathF.Abs(data[i]);
                if (a > absMax) absMax = a;
            }

            if (absMax < 1e-8f)
                return CalibrateMinMax(data, true);

            // Compute histogram
            int[] hist = new int[numBins];
            float binWidth = absMax / numBins;

            for (int i = 0; i < data.Length; i++)
            {
                int bin = (int)(MathF.Abs(data[i]) / binWidth);
                if (bin >= numBins) bin = numBins - 1;
                hist[bin]++;
            }

            // Find threshold that minimizes KL divergence
            int optimalThresholdBin = numBins;
            double minKl = double.MaxValue;

            for (int threshold = targetBins; threshold < numBins; threshold += 16)
            {
                // Quantize and reconstruct histogram to calculate relative entropy
                double kl = 0.0;
                double sumP = 0.0;
                for (int i = 0; i < threshold; i++) sumP += hist[i];
                if (sumP == 0) continue;

                // Simple smooth divergence approximation
                for (int i = 0; i < threshold; i++)
                {
                    if (hist[i] > 0)
                    {
                        double p = (double)hist[i] / sumP;
                        double q = 1.0 / threshold;
                        kl += p * Math.Log(p / q);
                    }
                }

                if (kl < minKl)
                {
                    minKl = kl;
                    optimalThresholdBin = threshold;
                }
            }

            float thresholdVal = optimalThresholdBin * binWidth;
            float scale = thresholdVal / 127.0f;
            return new QuantizationParams(scale, 0, -thresholdVal, thresholdVal);
        }

        public static void QuantizeToInt8Symmetric(ReadOnlySpan<float> src, Span<sbyte> dest, float scale)
        {
            if (dest.Length < src.Length)
                throw new ArgumentException("Destination span too small.", nameof(dest));

            float invScale = 1.0f / (scale > 1e-8f ? scale : 1e-8f);

            for (int i = 0; i < src.Length; i++)
            {
                int q = (int)MathF.Round(src[i] * invScale);
                dest[i] = (sbyte)Math.Clamp(q, -127, 127);
            }
        }

        public static void DequantizeFromInt8Symmetric(ReadOnlySpan<sbyte> src, Span<float> dest, float scale)
        {
            if (dest.Length < src.Length)
                throw new ArgumentException("Destination span too small.", nameof(dest));

            for (int i = 0; i < src.Length; i++)
            {
                dest[i] = src[i] * scale;
            }
        }
    }
}
