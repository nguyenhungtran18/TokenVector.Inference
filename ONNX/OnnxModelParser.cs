using System;
using System.IO;

namespace TokenVector.Inference.ONNX
{
    /// <summary>
    /// High-performance zero-copy ONNX Protobuf Binary Parser.
    /// </summary>
    public static class OnnxModelParser
    {
        public static OnnxModelProto Parse(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentNullException(nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"ONNX Model file not found: {filePath}", filePath);

            byte[] bytes = File.ReadAllBytes(filePath);
            return Parse(bytes);
        }

        public static OnnxModelProto Parse(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var model = new OnnxModelProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // ir_version
                        model.IrVersion = (long)reader.ReadVarint64();
                        break;
                    case 3: // producer_name
                        model.ProducerName = reader.ReadString();
                        break;
                    case 4: // producer_version
                        model.ProducerVersion = reader.ReadString();
                        break;
                    case 5: // domain
                        model.Domain = reader.ReadString();
                        break;
                    case 6: // model_version
                        model.ModelVersion = (long)reader.ReadVarint64();
                        break;
                    case 7: // doc_string
                        model.DocString = reader.ReadString();
                        break;
                    case 8: // graph
                        var graphSlice = reader.ReadLengthDelimited();
                        model.Graph = ParseGraph(graphSlice);
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return model;
        }

        private static OnnxGraphProto ParseGraph(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var graph = new OnnxGraphProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // node (NodeProto)
                        var nodeSlice = reader.ReadLengthDelimited();
                        graph.Nodes.Add(ParseNode(nodeSlice));
                        break;
                    case 2: // name
                        graph.Name = reader.ReadString();
                        break;
                    case 5: // initializer (TensorProto)
                        var tensorSlice = reader.ReadLengthDelimited();
                        graph.Initializers.Add(ParseTensor(tensorSlice));
                        break;
                    case 11: // input (ValueInfoProto)
                        var inputSlice = reader.ReadLengthDelimited();
                        graph.Inputs.Add(ParseValueInfo(inputSlice));
                        break;
                    case 12: // output (ValueInfoProto)
                        var outputSlice = reader.ReadLengthDelimited();
                        graph.Outputs.Add(ParseValueInfo(outputSlice));
                        break;
                    case 13: // value_info (ValueInfoProto)
                        var viSlice = reader.ReadLengthDelimited();
                        graph.ValueInfos.Add(ParseValueInfo(viSlice));
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return graph;
        }

        private static OnnxNodeProto ParseNode(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var node = new OnnxNodeProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // input
                        node.Inputs.Add(reader.ReadString());
                        break;
                    case 2: // output
                        node.Outputs.Add(reader.ReadString());
                        break;
                    case 3: // name
                        node.Name = reader.ReadString();
                        break;
                    case 4: // op_type
                        node.OpType = reader.ReadString();
                        break;
                    case 5: // attribute (AttributeProto)
                        var attrSlice = reader.ReadLengthDelimited();
                        var attr = ParseAttribute(attrSlice);
                        if (!string.IsNullOrEmpty(attr.Name))
                            node.Attributes[attr.Name] = attr;
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return node;
        }

        private static OnnxAttributeProto ParseAttribute(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var attr = new OnnxAttributeProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // name
                        attr.Name = reader.ReadString();
                        break;
                    case 2: // f (float)
                        attr.F = reader.ReadFloat();
                        break;
                    case 3: // i (int64)
                        attr.I = (long)reader.ReadVarint64();
                        break;
                    case 4: // s (string/bytes)
                        attr.S = reader.ReadString();
                        break;
                    case 5: // t (TensorProto)
                        var tensorSlice = reader.ReadLengthDelimited();
                        attr.T = ParseTensor(tensorSlice);
                        break;
                    case 7: // floats (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                attr.Floats.Add(subReader.ReadFloat());
                        }
                        else
                        {
                            attr.Floats.Add(reader.ReadFloat());
                        }
                        break;
                    case 8: // ints (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                attr.Ints.Add((long)subReader.ReadVarint64());
                        }
                        else
                        {
                            attr.Ints.Add((long)reader.ReadVarint64());
                        }
                        break;
                    case 9: // strings
                        attr.Strings.Add(reader.ReadString());
                        break;
                    case 20: // type
                        attr.Type = (AttributeType)reader.ReadVarint32();
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return attr;
        }

        private static OnnxTensorProto ParseTensor(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var tensor = new OnnxTensorProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // dims (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                tensor.Dims.Add((long)subReader.ReadVarint64());
                        }
                        else
                        {
                            tensor.Dims.Add((long)reader.ReadVarint64());
                        }
                        break;
                    case 2: // data_type
                        tensor.DataType = (TensorDataType)reader.ReadVarint32();
                        break;
                    case 4: // float_data (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                tensor.FloatData.Add(subReader.ReadFloat());
                        }
                        else
                        {
                            tensor.FloatData.Add(reader.ReadFloat());
                        }
                        break;
                    case 5: // int32_data (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                tensor.Int32Data.Add((int)subReader.ReadVarint32());
                        }
                        else
                        {
                            tensor.Int32Data.Add((int)reader.ReadVarint32());
                        }
                        break;
                    case 7: // int64_data (packed or repeated)
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                tensor.Int64Data.Add((long)subReader.ReadVarint64());
                        }
                        else
                        {
                            tensor.Int64Data.Add((long)reader.ReadVarint64());
                        }
                        break;
                    case 8: // name
                        tensor.Name = reader.ReadString();
                        break;
                    case 9: // raw_data
                        tensor.RawData = reader.ReadLengthDelimited().ToArray();
                        break;
                    case 10: // double_data
                        if (wireType == WireType.LengthDelimited)
                        {
                            var slice = reader.ReadLengthDelimited();
                            var subReader = new FastProtobufReader(slice);
                            while (subReader.HasMore)
                                tensor.DoubleData.Add(subReader.ReadDouble());
                        }
                        else
                        {
                            tensor.DoubleData.Add(reader.ReadDouble());
                        }
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return tensor;
        }

        private static OnnxValueInfoProto ParseValueInfo(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            var valueInfo = new OnnxValueInfoProto();

            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // name
                        valueInfo.Name = reader.ReadString();
                        break;
                    case 2: // type (TypeProto)
                        var typeSlice = reader.ReadLengthDelimited();
                        ParseTypeProto(typeSlice, valueInfo);
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return valueInfo;
        }

        private static void ParseTypeProto(ReadOnlySpan<byte> buffer, OnnxValueInfoProto target)
        {
            var reader = new FastProtobufReader(buffer);
            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                if (fieldNumber == 1) // tensor_type
                {
                    var tensorTypeSlice = reader.ReadLengthDelimited();
                    ParseTensorTypeProto(tensorTypeSlice, target);
                }
                else
                {
                    reader.SkipField(wireType);
                }
            }
        }

        private static void ParseTensorTypeProto(ReadOnlySpan<byte> buffer, OnnxValueInfoProto target)
        {
            var reader = new FastProtobufReader(buffer);
            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                switch (fieldNumber)
                {
                    case 1: // elem_type
                        target.DataType = (TensorDataType)reader.ReadVarint32();
                        break;
                    case 2: // shape (TensorShapeProto)
                        var shapeSlice = reader.ReadLengthDelimited();
                        ParseShapeProto(shapeSlice, target.Dims);
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }
        }

        private static void ParseShapeProto(ReadOnlySpan<byte> buffer, System.Collections.Generic.List<long> dims)
        {
            var reader = new FastProtobufReader(buffer);
            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                if (fieldNumber == 1) // dim (Dimension)
                {
                    var dimSlice = reader.ReadLengthDelimited();
                    long dimVal = ParseDimension(dimSlice);
                    dims.Add(dimVal);
                }
                else
                {
                    reader.SkipField(wireType);
                }
            }
        }

        private static long ParseDimension(ReadOnlySpan<byte> buffer)
        {
            var reader = new FastProtobufReader(buffer);
            long val = 1;
            while (reader.ReadTag(out int fieldNumber, out WireType wireType))
            {
                if (fieldNumber == 1) // dim_value
                {
                    val = (long)reader.ReadVarint64();
                }
                else
                {
                    reader.SkipField(wireType);
                }
            }
            return val;
        }
    }
}
