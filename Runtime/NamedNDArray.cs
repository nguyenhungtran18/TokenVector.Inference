using System;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Runtime
{
    /// <summary>
    /// Lightweight zero-allocation struct holding a named tensor input or output.
    /// </summary>
    public readonly struct NamedNDArray
    {
        public readonly string Name;
        public readonly NDArray<float>? FloatArray;
        public readonly ReadOnlyMemory<float> Memory;

        public NamedNDArray(string name, NDArray<float> array)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            FloatArray = array ?? throw new ArgumentNullException(nameof(array));
            Memory = default;
        }

        public NamedNDArray(string name, ReadOnlyMemory<float> memory)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            FloatArray = null;
            Memory = memory;
        }

        public readonly ReadOnlySpan<float> AsReadOnlySpan()
        {
            if (FloatArray != null)
                return FloatArray.AsSpan();
            return Memory.Span;
        }

        public readonly Span<float> AsSpan()
        {
            if (FloatArray != null)
                return FloatArray.AsSpan();
            throw new InvalidOperationException("Cannot obtain mutable Span from ReadOnlyMemory NamedNDArray.");
        }

        public readonly int[] GetShape()
        {
            if (FloatArray != null)
                return FloatArray.Shape;
            return new int[] { Memory.Length };
        }
    }
}
