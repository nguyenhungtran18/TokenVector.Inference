using System;
using System.Collections.Generic;

namespace TokenVector.Inference.ONNX
{
    public enum TensorDataType
    {
        Undefined = 0,
        Float = 1,
        Uint8 = 2,
        Int8 = 3,
        Uint16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        Uint32 = 12,
        Uint64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        BFloat16 = 16,
        Float8E4M3FN = 17,
        Float8E4M3FNUZ = 18,
        Float8E5M2 = 19,
        Float8E5M2FNUZ = 20
    }

    public enum AttributeType
    {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
        SparseTensor = 11,
        SparseTensors = 12,
        TypeProto = 13,
        TypeProtos = 14
    }

    public sealed class OnnxAttributeProto
    {
        public string Name { get; set; } = string.Empty;
        public AttributeType Type { get; set; } = AttributeType.Undefined;
        public float F { get; set; }
        public long I { get; set; }
        public string S { get; set; } = string.Empty;
        public OnnxTensorProto? T { get; set; }
        public List<float> Floats { get; } = new();
        public List<long> Ints { get; } = new();
        public List<string> Strings { get; } = new();
    }

    public sealed class OnnxTensorProto
    {
        public string Name { get; set; } = string.Empty;
        public TensorDataType DataType { get; set; } = TensorDataType.Float;
        public List<long> Dims { get; } = new();
        public byte[]? RawData { get; set; }
        public List<float> FloatData { get; } = new();
        public List<int> Int32Data { get; } = new();
        public List<long> Int64Data { get; } = new();
        public List<double> DoubleData { get; } = new();

        public float[] ExtractFloatArray()
        {
            if (FloatData.Count > 0)
                return FloatData.ToArray();

            if (RawData != null && RawData.Length >= 4)
            {
                int count = RawData.Length / 4;
                float[] result = new float[count];
                Buffer.BlockCopy(RawData, 0, result, 0, RawData.Length);
                return result;
            }

            return Array.Empty<float>();
        }

        public int[] ExtractInt32Array()
        {
            if (Int32Data.Count > 0)
                return Int32Data.ToArray();

            if (Int64Data.Count > 0)
            {
                int[] res = new int[Int64Data.Count];
                for (int i = 0; i < Int64Data.Count; i++)
                    res[i] = (int)Int64Data[i];
                return res;
            }

            if (RawData != null && RawData.Length >= 4)
            {
                int count = RawData.Length / 4;
                int[] result = new int[count];
                Buffer.BlockCopy(RawData, 0, result, 0, RawData.Length);
                return result;
            }

            return Array.Empty<int>();
        }
    }

    public sealed class OnnxValueInfoProto
    {
        public string Name { get; set; } = string.Empty;
        public TensorDataType DataType { get; set; } = TensorDataType.Float;
        public List<long> Dims { get; } = new();
    }

    public sealed class OnnxNodeProto
    {
        public string Name { get; set; } = string.Empty;
        public string OpType { get; set; } = string.Empty;
        public List<string> Inputs { get; } = new();
        public List<string> Outputs { get; } = new();
        public Dictionary<string, OnnxAttributeProto> Attributes { get; } = new(StringComparer.Ordinal);
    }

    public sealed class OnnxGraphProto
    {
        public string Name { get; set; } = string.Empty;
        public List<OnnxNodeProto> Nodes { get; } = new();
        public List<OnnxTensorProto> Initializers { get; } = new();
        public List<OnnxValueInfoProto> Inputs { get; } = new();
        public List<OnnxValueInfoProto> Outputs { get; } = new();
        public List<OnnxValueInfoProto> ValueInfos { get; } = new();
    }

    public sealed class OnnxModelProto
    {
        public long IrVersion { get; set; }
        public string ProducerName { get; set; } = string.Empty;
        public string ProducerVersion { get; set; } = string.Empty;
        public string Domain { get; set; } = string.Empty;
        public long ModelVersion { get; set; }
        public string DocString { get; set; } = string.Empty;
        public OnnxGraphProto Graph { get; set; } = new();
    }
}
