using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TokenVector.Inference.Runtime
{
    /// <summary>
    /// Linear unmanaged memory scratchpad arena aligned to 64-byte boundaries for AVX2/AVX-512 SIMD kernels.
    /// Provides zero-allocation tensor slices throughout the entire inference hot-path.
    /// </summary>
    public sealed unsafe class ExecutionMemoryArena : IDisposable
    {
        private byte* _basePtr;
        private readonly nuint _sizeInBytes;
        private bool _disposed;

        public ExecutionMemoryArena(nuint sizeInBytes)
        {
            if (sizeInBytes == 0)
                sizeInBytes = 64; // Minimum non-zero allocation

            // Align total size to 64 bytes
            sizeInBytes = (sizeInBytes + 63) & ~((nuint)63);
            _sizeInBytes = sizeInBytes;
            _basePtr = (byte*)NativeMemory.AlignedAlloc(sizeInBytes, 64);
            NativeMemory.Clear(_basePtr, sizeInBytes);
        }

        public byte* BasePointer => _basePtr;
        public nuint SizeInBytes => _sizeInBytes;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<float> GetFloatSpan(int byteOffset, int count)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ExecutionMemoryArena));
            if (byteOffset < 0 || (nuint)(byteOffset + count * sizeof(float)) > _sizeInBytes)
                throw new ArgumentOutOfRangeException(nameof(byteOffset), $"Offset {byteOffset} with {count} floats exceeds arena size {_sizeInBytes}.");

            return new Span<float>(_basePtr + byteOffset, count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetReadOnlyFloatSpan(int byteOffset, int count)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ExecutionMemoryArena));
            if (byteOffset < 0 || (nuint)(byteOffset + count * sizeof(float)) > _sizeInBytes)
                throw new ArgumentOutOfRangeException(nameof(byteOffset), $"Offset {byteOffset} with {count} floats exceeds arena size {_sizeInBytes}.");

            return new ReadOnlySpan<float>(_basePtr + byteOffset, count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<byte> GetByteSpan(int byteOffset, int count)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ExecutionMemoryArena));
            if (byteOffset < 0 || (nuint)(byteOffset + count) > _sizeInBytes)
                throw new ArgumentOutOfRangeException(nameof(byteOffset), $"Offset {byteOffset} with {count} bytes exceeds arena size {_sizeInBytes}.");

            return new Span<byte>(_basePtr + byteOffset, count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<sbyte> GetInt8Span(int byteOffset, int count)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ExecutionMemoryArena));
            if (byteOffset < 0 || (nuint)(byteOffset + count) > _sizeInBytes)
                throw new ArgumentOutOfRangeException(nameof(byteOffset), $"Offset {byteOffset} with {count} int8 elements exceeds arena size {_sizeInBytes}.");

            return new Span<sbyte>(_basePtr + byteOffset, count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void CopyFrom(int byteOffset, ReadOnlySpan<float> source)
        {
            Span<float> dest = GetFloatSpan(byteOffset, source.Length);
            source.CopyTo(dest);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void CopyTo(int byteOffset, Span<float> destination)
        {
            ReadOnlySpan<float> src = GetReadOnlyFloatSpan(byteOffset, destination.Length);
            src.CopyTo(destination);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_basePtr != null)
                {
                    NativeMemory.AlignedFree(_basePtr);
                    _basePtr = null;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~ExecutionMemoryArena()
        {
            Dispose();
        }
    }
}
