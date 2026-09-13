using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Inference.Quantization
{
    /// <summary>
    /// Implementation of 8-bit floating point formats (FP8):
    /// - E4M3 (1 sign, 4 exponent bits, 3 mantissa bits, bias = 7): High precision for weights and activations.
    /// - E5M2 (1 sign, 5 exponent bits, 2 mantissa bits, bias = 15): High dynamic range for gradients / attention logits.
    /// </summary>
    public readonly struct FP8E4M3 : IEquatable<FP8E4M3>
    {
        public readonly byte Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FP8E4M3(byte value) => Value = value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static FP8E4M3 FromFloat(float val)
        {
            if (float.IsNaN(val)) return new FP8E4M3(0x7F); // Canonical NaN
            uint u = BitConverter.SingleToUInt32Bits(val);
            uint sign = (u >> 31) & 0x1;
            int exp = (int)((u >> 23) & 0xFF) - 127 + 7; // Convert bias 127 -> 7
            uint mant = (u >> 20) & 0x7; // 3 bits

            if (exp >= 15) // Max exponent
            {
                return new FP8E4M3((byte)((sign << 7) | 0x7E)); // Saturation max normal
            }
            if (exp <= 0) // Subnormal or zero
            {
                if (exp < -3) return new FP8E4M3((byte)(sign << 7)); // Zero
                mant |= 0x8;
                mant >>= (1 - exp);
                return new FP8E4M3((byte)((sign << 7) | ((uint)mant & 0x7)));
            }

            return new FP8E4M3((byte)((sign << 7) | (((uint)exp & 0xF) << 3) | ((uint)mant & 0x7)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ToFloat()
        {
            uint sign = (uint)((Value >> 7) & 0x1);
            int exp = (Value >> 3) & 0xF;
            uint mant = (uint)(Value & 0x7);

            if (exp == 0xF && mant == 0x7) return float.NaN;

            if (exp == 0)
            {
                if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
                // Subnormal
                float sub = (float)mant / 8.0f * MathF.Pow(2, -6);
                return sign == 1 ? -sub : sub;
            }

            float normal = (1.0f + (float)mant / 8.0f) * MathF.Pow(2, exp - 7);
            return sign == 1 ? -normal : normal;
        }

        public bool Equals(FP8E4M3 other) => Value == other.Value;
        public override bool Equals(object? obj) => obj is FP8E4M3 other && Equals(other);
        public override int GetHashCode() => Value.GetHashCode();
        public override string ToString() => ToFloat().ToString("G4");
    }

    public readonly struct FP8E5M2 : IEquatable<FP8E5M2>
    {
        public readonly byte Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FP8E5M2(byte value) => Value = value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static FP8E5M2 FromFloat(float val)
        {
            if (float.IsNaN(val)) return new FP8E5M2(0x7F);
            if (float.IsPositiveInfinity(val)) return new FP8E5M2(0x7C);
            if (float.IsNegativeInfinity(val)) return new FP8E5M2(0xFC);

            uint u = BitConverter.SingleToUInt32Bits(val);
            uint sign = (u >> 31) & 0x1;
            int exp = (int)((u >> 23) & 0xFF) - 127 + 15; // Convert bias 127 -> 15
            uint mant = (u >> 21) & 0x3; // 2 bits

            if (exp >= 31)
            {
                return new FP8E5M2((byte)((sign << 7) | 0x7C)); // Infinity
            }
            if (exp <= 0)
            {
                if (exp < -2) return new FP8E5M2((byte)(sign << 7));
                mant |= 0x4;
                mant >>= (1 - exp);
                return new FP8E5M2((byte)((sign << 7) | ((uint)mant & 0x3)));
            }

            return new FP8E5M2((byte)((sign << 7) | (((uint)exp & 0x1F) << 2) | ((uint)mant & 0x3)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ToFloat()
        {
            uint sign = (uint)((Value >> 7) & 0x1);
            int exp = (Value >> 2) & 0x1F;
            uint mant = (uint)(Value & 0x3);

            if (exp == 0x1F)
            {
                if (mant == 0) return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
                return float.NaN;
            }

            if (exp == 0)
            {
                if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
                float sub = (float)mant / 4.0f * MathF.Pow(2, -14);
                return sign == 1 ? -sub : sub;
            }

            float normal = (1.0f + (float)mant / 4.0f) * MathF.Pow(2, exp - 15);
            return sign == 1 ? -normal : normal;
        }

        public bool Equals(FP8E5M2 other) => Value == other.Value;
        public override bool Equals(object? obj) => obj is FP8E5M2 other && Equals(other);
        public override int GetHashCode() => Value.GetHashCode();
        public override string ToString() => ToFloat().ToString("G4");
    }
}
