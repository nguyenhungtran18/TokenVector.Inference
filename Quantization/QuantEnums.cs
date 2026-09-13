using System;

namespace TokenVector.Inference.Quantization
{
    public enum QuantizationPrecision
    {
        FP32 = 0,
        FP16 = 1,
        INT8_Symmetric = 2,
        INT8_Asymmetric = 3,
        FP8_E4M3 = 4,
        FP8_E5M2 = 5
    }

    public enum CalibrationMethod
    {
        MinMax = 0,
        KLDivergence = 1,
        Percentile = 2
    }

    public readonly struct QuantizationParams
    {
        public readonly float Scale;
        public readonly int ZeroPoint;
        public readonly float MinVal;
        public readonly float MaxVal;

        public QuantizationParams(float scale, int zeroPoint, float minVal, float maxVal)
        {
            Scale = scale;
            ZeroPoint = zeroPoint;
            MinVal = minVal;
            MaxVal = maxVal;
        }

        public static QuantizationParams Identity => new(1.0f, 0, float.MinValue, float.MaxValue);
    }
}
