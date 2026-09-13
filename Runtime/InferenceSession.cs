using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using TokenVector.Inference.ONNX;
using TokenVector.Inference.Optimization;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Runtime
{
    /// <summary>
    /// High-performance Deep Learning Inference Engine with Zero-GC Allocation Hot Path.
    /// Executes optimized execution graphs directly against pre-allocated unmanaged memory arenas.
    /// </summary>
    public sealed class InferenceSession : IDisposable
    {
        private readonly ExecutionGraph _graph;
        private readonly ExecutionMemoryArena _arena;
        private readonly List<IGraphPass> _passes = new();
        private bool _disposed;

        public InferenceSession(string onnxFilePath)
            : this(OnnxModelParser.Parse(onnxFilePath))
        {
        }

        public InferenceSession(ReadOnlySpan<byte> onnxModelBytes)
            : this(OnnxModelParser.Parse(onnxModelBytes))
        {
        }

        public InferenceSession(OnnxModelProto modelProto)
            : this(GraphTranslator.Translate(modelProto))
        {
        }

        public InferenceSession(ExecutionGraph graph, bool applyOptimizations = true)
        {
            _graph = graph ?? throw new ArgumentNullException(nameof(graph));

            if (applyOptimizations)
            {
                _passes.Add(new ConstantFoldingPass());
                _passes.Add(new OperatorFusionPass());
                _passes.Add(new DeadCodeEliminationPass());

                foreach (var pass in _passes)
                {
                    pass.Run(_graph);
                }
            }

            // Perform tensor liveness analysis & calculate minimum scratchpad bytes
            _graph.AnalyzeLivenessAndAllocateOffsets();

            // Allocate unified native memory arena
            _arena = new ExecutionMemoryArena(_graph.TotalArenaBytes);

            // Populate persistent weights
            _graph.InitializeWeights(_arena);
        }

        public ExecutionGraph Graph => _graph;
        public nuint TotalMemoryAllocatedBytes => _arena.SizeInBytes;

        /// <summary>
        /// ZERO-ALLOCATION HOT PATH: Runs inference without allocating any memory on the managed heap.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Run(ReadOnlySpan<NamedNDArray> inputs, Span<NamedNDArray> outputs)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));

            // 1. Copy inputs into designated arena slots
            for (int i = 0; i < inputs.Length; i++)
            {
                ref readonly var inp = ref inputs[i];
                if (_graph.Tensors.TryGetValue(inp.Name, out var desc))
                {
                    _arena.CopyFrom(desc.Offset, inp.AsReadOnlySpan());
                }
            }

            // 2. Execute topologically sorted nodes
            var nodes = _graph.Nodes;
            int nodeCount = nodes.Count;
            for (int i = 0; i < nodeCount; i++)
            {
                nodes[i].Execute(_arena);
            }

            // 3. Copy outputs from designated arena slots
            for (int i = 0; i < outputs.Length; i++)
            {
                ref var outp = ref outputs[i];
                if (_graph.Tensors.TryGetValue(outp.Name, out var desc))
                {
                    _arena.CopyTo(desc.Offset, outp.AsSpan());
                }
            }
        }

        /// <summary>
        /// Single-tensor inference helper.
        /// </summary>
        public NDArray<float> Run(NDArray<float> input, string? inputName = null, string? outputName = null)
        {
            string inName = inputName ?? (_graph.Inputs.Count > 0 ? _graph.Inputs[0] : "input");
            string outName = outputName ?? (_graph.Outputs.Count > 0 ? _graph.Outputs[0] : "output");

            if (!_graph.Tensors.TryGetValue(outName, out var outDesc))
                throw new InvalidOperationException($"Output tensor '{outName}' not found in graph.");

            var outputArray = new NDArray<float>(outDesc.Shape.Length > 0 ? outDesc.Shape : new int[] { outDesc.ElementCount });

            NamedNDArray[] inArray = new NamedNDArray[1] { new(inName, input) };
            NamedNDArray[] outArrayWrapper = new NamedNDArray[1] { new(outName, outputArray) };

            Run(inArray, outArrayWrapper);

            return outputArray;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _arena.Dispose();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~InferenceSession()
        {
            Dispose();
        }
    }
}
