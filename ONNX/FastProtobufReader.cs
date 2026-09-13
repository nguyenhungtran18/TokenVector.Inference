using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace TokenVector.Inference.ONNX
{
    /// <summary>
    /// Ultra-fast zero-allocation binary Protobuf wire reader operating directly on ReadOnlySpan&lt;byte&gt;.
    /// </summary>
    public ref struct FastProtobufReader
    {
        private ReadOnlySpan<byte> _buffer;
        private int _position;

        public FastProtobufReader(ReadOnlySpan<byte> buffer)
        {
            _buffer = buffer;
            _position = 0;
        }

        public readonly int Position => _position;
        public readonly int Length => _buffer.Length;
        public readonly bool HasMore => _position < _buffer.Length;
        public readonly int Remaining => _buffer.Length - _position;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool ReadTag(out int fieldNumber, out WireType wireType)
        {
            if (_position >= _buffer.Length)
            {
                fieldNumber = 0;
                wireType = WireType.Varint;
                return false;
            }

            uint tag = ReadVarint32();
            fieldNumber = (int)(tag >> 3);
            wireType = (WireType)(tag & 0x07);
            return fieldNumber > 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint ReadVarint32()
        {
            uint result = 0;
            int shift = 0;

            while (_position < _buffer.Length && shift < 35)
            {
                byte b = _buffer[_position++];
                result |= (uint)(b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                {
                    return result;
                }
                shift += 7;
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong ReadVarint64()
        {
            ulong result = 0;
            int shift = 0;

            while (_position < _buffer.Length && shift < 70)
            {
                byte b = _buffer[_position++];
                result |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                {
                    return result;
                }
                shift += 7;
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ReadFloat()
        {
            if (_position + 4 > _buffer.Length)
                throw new InvalidOperationException("Unexpected end of buffer reading Float32.");

            float val = BitConverter.ToSingle(_buffer.Slice(_position, 4));
            _position += 4;
            return val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double ReadDouble()
        {
            if (_position + 8 > _buffer.Length)
                throw new InvalidOperationException("Unexpected end of buffer reading Float64.");

            double val = BitConverter.ToDouble(_buffer.Slice(_position, 8));
            _position += 8;
            return val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint ReadFixed32()
        {
            if (_position + 4 > _buffer.Length)
                throw new InvalidOperationException("Unexpected end of buffer reading Fixed32.");

            uint val = BitConverter.ToUInt32(_buffer.Slice(_position, 4));
            _position += 4;
            return val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong ReadFixed64()
        {
            if (_position + 8 > _buffer.Length)
                throw new InvalidOperationException("Unexpected end of buffer reading Fixed64.");

            ulong val = BitConverter.ToUInt64(_buffer.Slice(_position, 8));
            _position += 8;
            return val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> ReadLengthDelimited()
        {
            int length = (int)ReadVarint32();
            if (length < 0 || _position + length > _buffer.Length)
                throw new InvalidOperationException($"Invalid length delimited segment: {length} bytes.");

            ReadOnlySpan<byte> slice = _buffer.Slice(_position, length);
            _position += length;
            return slice;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string ReadString()
        {
            ReadOnlySpan<byte> bytes = ReadLengthDelimited();
            return bytes.Length == 0 ? string.Empty : Encoding.UTF8.GetString(bytes);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SkipField(WireType wireType)
        {
            switch (wireType)
            {
                case WireType.Varint:
                    ReadVarint64();
                    break;
                case WireType.Fixed64:
                    _position += 8;
                    break;
                case WireType.LengthDelimited:
                    int len = (int)ReadVarint32();
                    _position += len;
                    break;
                case WireType.Fixed32:
                    _position += 4;
                    break;
                default:
                    throw new NotSupportedException($"Unsupported wire type: {wireType}");
            }
        }
    }

    public enum WireType
    {
        Varint = 0,
        Fixed64 = 1,
        LengthDelimited = 2,
        StartGroup = 3,
        EndGroup = 4,
        Fixed32 = 5
    }
}
