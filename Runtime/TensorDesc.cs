using System;
using TokenVector.Inference.ONNX;

namespace TokenVector.Inference.Runtime
{
    /// <summary>
    /// Metadata descriptor for a tensor in the execution graph, including shape, data type, and memory allocation slot.
    /// </summary>
    public sealed class TensorDesc
    {
        public string Name { get; set; } = string.Empty;
        public TensorDataType DataType { get; set; } = TensorDataType.Float;
        public int[] Shape { get; set; } = Array.Empty<int>();
        public int ElementCount { get; set; }
        public int ByteSize { get; set; }
        public int Offset { get; set; } = -1;
        public bool IsInput { get; set; }
        public bool IsOutput { get; set; }
        public bool IsInitializer { get; set; }
        public int LifetimeStart { get; set; } = -1;
        public int LifetimeEnd { get; set; } = -1;

        public TensorDesc() { }

        public TensorDesc(string name, int[]? shape, TensorDataType dataType = TensorDataType.Float)
        {
            Name = name;
            int[] actualShape = shape ?? Array.Empty<int>();
            Shape = actualShape;
            DataType = dataType;
            ElementCount = CalculateElementCount(actualShape);
            ByteSize = ElementCount * GetTypeSize(dataType);
        }

        public static int CalculateElementCount(int[]? shape)
        {
            if (shape == null || shape.Length == 0) return 1;
            int total = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                int dim = shape[i] > 0 ? shape[i] : 1;
                total *= dim;
            }
            return total;
        }

        public static int GetTypeSize(TensorDataType type)
        {
            return type switch
            {
                TensorDataType.Float => 4,
                TensorDataType.Float16 => 2,
                TensorDataType.BFloat16 => 2,
                TensorDataType.Int8 => 1,
                TensorDataType.Uint8 => 1,
                TensorDataType.Float8E4M3FN => 1,
                TensorDataType.Float8E5M2 => 1,
                TensorDataType.Int32 => 4,
                TensorDataType.Int64 => 8,
                TensorDataType.Double => 8,
                _ => 4
            };
        }
    }
}
