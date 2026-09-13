using System;
using System.IO;
using System.Text;
using TokenVector.Inference.ONNX;
using Xunit;

namespace TokenVector.Inference.Tests
{
    public class FastProtobufReaderTests
    {
        [Fact]
        public void ReadVarint32_CorrectlyDecodesValues()
        {
            // Test 1-byte varint (150 -> 0x96 0x01)
            byte[] bytes = new byte[] { 0x96, 0x01 };
            var reader = new FastProtobufReader(bytes);
            uint val = reader.ReadVarint32();
            Assert.Equal(150u, val);
        }

        [Fact]
        public void ReadVarint64_CorrectlyDecodesLargeValues()
        {
            // 300 -> 0xAC 0x02
            byte[] bytes = new byte[] { 0xAC, 0x02 };
            var reader = new FastProtobufReader(bytes);
            ulong val = reader.ReadVarint64();
            Assert.Equal(300ul, val);
        }

        [Fact]
        public void ReadFloat_CorrectlyDecodesIEEE754()
        {
            float expected = 3.1415927f;
            byte[] bytes = BitConverter.GetBytes(expected);
            var reader = new FastProtobufReader(bytes);
            float actual = reader.ReadFloat();
            Assert.Equal(expected, actual, precision: 6);
        }

        [Fact]
        public void ReadLengthDelimitedString_ReturnsExpectedString()
        {
            string expected = "TokenVector.Inference";
            byte[] strBytes = Encoding.UTF8.GetBytes(expected);
            using var ms = new MemoryStream();
            ms.WriteByte((byte)strBytes.Length);
            ms.Write(strBytes);

            var reader = new FastProtobufReader(ms.ToArray());
            string actual = reader.ReadString();
            Assert.Equal(expected, actual);
        }
    }
}
