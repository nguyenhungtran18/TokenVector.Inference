using System;
using System.Collections.Generic;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Runtime
{
    /// <summary>
    /// Topologically sorted execution graph with Tensor Liveness Analysis and automated memory scratchpad reuse.
    /// </summary>
    public sealed class ExecutionGraph
    {
        public string Name { get; set; } = "TokenVectorGraph";
        public List<ExecutionNode> Nodes { get; } = new();
        public Dictionary<string, TensorDesc> Tensors { get; } = new(StringComparer.Ordinal);
        public List<string> Inputs { get; } = new();
        public List<string> Outputs { get; } = new();
        public Dictionary<string, NDArray<float>> Initializers { get; } = new(StringComparer.Ordinal);

        public nuint TotalArenaBytes { get; private set; }

        public TensorDesc GetOrCreateTensor(string name, int[] shape, ONNX.TensorDataType dataType = ONNX.TensorDataType.Float)
        {
            if (Tensors.TryGetValue(name, out var existing))
            {
                if (shape.Length > 0 && existing.Shape.Length == 0)
                {
                    existing.Shape = shape;
                    existing.ElementCount = TensorDesc.CalculateElementCount(shape);
                    existing.ByteSize = existing.ElementCount * TensorDesc.GetTypeSize(dataType);
                }
                return existing;
            }

            var desc = new TensorDesc(name, shape, dataType);
            Tensors[name] = desc;
            return desc;
        }

        /// <summary>
        /// Analyzes tensor lifetime intervals and allocates compact memory offsets with buffer reuse.
        /// </summary>
        public void AnalyzeLivenessAndAllocateOffsets()
        {
            // 1. Determine lifetimes for all tensors
            for (int i = 0; i < Nodes.Count; i++)
            {
                var node = Nodes[i];
                node.Id = i;

                // Bind input descs
                node.InputDescs.Clear();
                foreach (var inName in node.Inputs)
                {
                    if (Tensors.TryGetValue(inName, out var t))
                    {
                        node.InputDescs.Add(t);
                        if (t.LifetimeStart == -1) t.LifetimeStart = i;
                        t.LifetimeEnd = i;
                    }
                }

                // Bind output descs
                node.OutputDescs.Clear();
                foreach (var outName in node.Outputs)
                {
                    if (Tensors.TryGetValue(outName, out var t))
                    {
                        node.OutputDescs.Add(t);
                        if (t.LifetimeStart == -1) t.LifetimeStart = i;
                        t.LifetimeEnd = i;
                    }
                }
            }

            // Inputs and outputs must remain alive throughout the session run
            foreach (var inp in Inputs)
            {
                if (Tensors.TryGetValue(inp, out var t))
                {
                    t.IsInput = true;
                    t.LifetimeStart = 0;
                    t.LifetimeEnd = Nodes.Count;
                }
            }
            foreach (var outp in Outputs)
            {
                if (Tensors.TryGetValue(outp, out var t))
                {
                    t.IsOutput = true;
                    t.LifetimeStart = 0;
                    t.LifetimeEnd = Nodes.Count;
                }
            }

            // 2. Allocate persistent segment for Initializers
            int currentOffset = 0;
            foreach (var (name, arr) in Initializers)
            {
                if (Tensors.TryGetValue(name, out var t))
                {
                    t.IsInitializer = true;
                    t.Offset = currentOffset;
                    int alignedSize = (t.ByteSize + 63) & ~63;
                    currentOffset += alignedSize;
                }
            }

            int persistentOffset = currentOffset;

            // 3. Allocate persistent segment for Model Inputs and Model Outputs
            foreach (var inp in Inputs)
            {
                if (Tensors.TryGetValue(inp, out var t) && t.Offset == -1)
                {
                    t.Offset = currentOffset;
                    int alignedSize = (t.ByteSize + 63) & ~63;
                    currentOffset += alignedSize;
                }
            }
            foreach (var outp in Outputs)
            {
                if (Tensors.TryGetValue(outp, out var t) && t.Offset == -1)
                {
                    t.Offset = currentOffset;
                    int alignedSize = (t.ByteSize + 63) & ~63;
                    currentOffset += alignedSize;
                }
            }

            // 4. Scratchpad buffer reuse for intermediate tensors (Greedy Interval Allocation)
            var intermediateTensors = new List<TensorDesc>();
            foreach (var t in Tensors.Values)
            {
                if (!t.IsInitializer && !t.IsInput && !t.IsOutput)
                {
                    intermediateTensors.Add(t);
                }
            }

            // Sort by lifetime start
            intermediateTensors.Sort((a, b) => a.LifetimeStart.CompareTo(b.LifetimeStart));

            var activeSlots = new List<(int offset, int size, int lifetimeEnd)>();
            int maxScratchpadBytes = 0;

            foreach (var t in intermediateTensors)
            {
                // Free slots that ended before t.LifetimeStart
                activeSlots.RemoveAll(slot => slot.lifetimeEnd < t.LifetimeStart);

                // Try to find a reusable slot with sufficient size
                int chosenOffset = -1;
                for (int s = 0; s < activeSlots.Count; s++)
                {
                    if (activeSlots[s].size >= t.ByteSize)
                    {
                        chosenOffset = activeSlots[s].offset;
                        activeSlots[s] = (chosenOffset, activeSlots[s].size, t.LifetimeEnd);
                        break;
                    }
                }

                if (chosenOffset == -1)
                {
                    // Allocate new slot at the end of scratchpad
                    chosenOffset = currentOffset + maxScratchpadBytes;
                    int alignedSize = (t.ByteSize + 63) & ~63;
                    maxScratchpadBytes += alignedSize;
                    activeSlots.Add((chosenOffset, alignedSize, t.LifetimeEnd));
                }

                t.Offset = chosenOffset;
            }

            TotalArenaBytes = (nuint)(currentOffset + maxScratchpadBytes + 64);
        }

        public unsafe void InitializeWeights(ExecutionMemoryArena arena)
        {
            foreach (var (name, arr) in Initializers)
            {
                if (Tensors.TryGetValue(name, out var t) && t.Offset >= 0)
                {
                    ReadOnlySpan<float> span = arr.AsSpan();
                    arena.CopyFrom(t.Offset, span);
                }
            }
        }
    }
}
