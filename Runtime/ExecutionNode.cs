using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Inference.Optimization;
using TokenVector.Inference.Quantization;

namespace TokenVector.Inference.Runtime
{
    public enum OperatorType
    {
        Custom = 0,
        MatMul = 1,
        FusedLinear = 2,
        Conv2D = 3,
        FusedConv2D = 4,
        RMSNorm = 5,
        LayerNorm = 6,
        Attention = 7,
        Relu = 8,
        Gelu = 9,
        Silu = 10,
        Sigmoid = 11,
        Tanh = 12,
        Add = 13,
        Sub = 14,
        Mul = 15,
        Div = 16,
        Softmax = 17,
        Reshape = 18,
        Transpose = 19,
        Concat = 20,
        QuantizedGemm = 21,
        Identity = 22
    }

    /// <summary>
    /// Node in the execution graph. Operates directly on the unmanaged memory arena with zero allocations.
    /// </summary>
    public sealed class ExecutionNode
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public OperatorType OpType { get; set; }
        public List<string> Inputs { get; } = new();
        public List<string> Outputs { get; } = new();

        // Memory slot descriptors
        public List<TensorDesc> InputDescs { get; } = new();
        public List<TensorDesc> OutputDescs { get; } = new();

        // Operator Attributes
        public ActivationFunction Activation { get; set; } = ActivationFunction.None;
        public float Alpha { get; set; } = 1.0f;
        public float Beta { get; set; } = 1.0f;
        public float Epsilon { get; set; } = 1e-5f;
        public int TransA { get; set; } = 0;
        public int TransB { get; set; } = 0;
        public int[] Strides { get; set; } = new int[] { 1, 1 };
        public int[] Pads { get; set; } = new int[] { 0, 0, 0, 0 };
        public int[] Perm { get; set; } = Array.Empty<int>();
        public int Axis { get; set; } = -1;

        // Quantization properties
        public float ScaleA { get; set; } = 1.0f;
        public float ScaleB { get; set; } = 1.0f;
        public float ScaleY { get; set; } = 1.0f;
        public int ZeroPointA { get; set; } = 0;
        public int ZeroPointB { get; set; } = 0;
        public int ZeroPointY { get; set; } = 0;

        public unsafe void Execute(ExecutionMemoryArena arena)
        {
            switch (OpType)
            {
                case OperatorType.FusedLinear:
                case OperatorType.MatMul:
                    ExecuteLinear(arena);
                    break;
                case OperatorType.FusedConv2D:
                case OperatorType.Conv2D:
                    ExecuteConv2D(arena);
                    break;
                case OperatorType.RMSNorm:
                    ExecuteRMSNorm(arena);
                    break;
                case OperatorType.LayerNorm:
                    ExecuteLayerNorm(arena);
                    break;
                case OperatorType.Relu:
                    ExecuteUnaryElementwise(arena, ActivationFunction.ReLU);
                    break;
                case OperatorType.Gelu:
                    ExecuteUnaryElementwise(arena, ActivationFunction.GELU);
                    break;
                case OperatorType.Silu:
                    ExecuteUnaryElementwise(arena, ActivationFunction.SiLU);
                    break;
                case OperatorType.Sigmoid:
                    ExecuteUnaryElementwise(arena, ActivationFunction.Sigmoid);
                    break;
                case OperatorType.Tanh:
                    ExecuteUnaryElementwise(arena, ActivationFunction.Tanh);
                    break;
                case OperatorType.Add:
                    ExecuteBinaryElementwise(arena, (a, b) => a + b);
                    break;
                case OperatorType.Sub:
                    ExecuteBinaryElementwise(arena, (a, b) => a - b);
                    break;
                case OperatorType.Mul:
                    ExecuteBinaryElementwise(arena, (a, b) => a * b);
                    break;
                case OperatorType.Div:
                    ExecuteBinaryElementwise(arena, (a, b) => a / b);
                    break;
                case OperatorType.Softmax:
                    ExecuteSoftmax(arena);
                    break;
                case OperatorType.Attention:
                    ExecuteAttention(arena);
                    break;
                case OperatorType.QuantizedGemm:
                    ExecuteQuantizedGemm(arena);
                    break;
                case OperatorType.Reshape:
                case OperatorType.Identity:
                    ExecuteIdentityOrReshape(arena);
                    break;
                case OperatorType.Transpose:
                    ExecuteTranspose(arena);
                    break;
                default:
                    ExecuteIdentityOrReshape(arena);
                    break;
            }
        }

        private unsafe void ExecuteLinear(ExecutionMemoryArena arena)
        {
            var inA = InputDescs[0];
            var inB = InputDescs[1];
            var outY = OutputDescs[0];

            int M = inA.Shape.Length > 1 ? inA.Shape[0] : 1;
            int K = inA.Shape.Length > 1 ? inA.Shape[^1] : inA.ElementCount;

            // Handle multi-dimensional input (e.g. [batch, seq, hidden])
            if (inA.Shape.Length > 2)
            {
                M = 1;
                for (int i = 0; i < inA.Shape.Length - 1; i++) M *= inA.Shape[i];
                K = inA.Shape[^1];
            }

            bool transB = TransB != 0;
            int N;
            if (inB.Shape.Length == 2)
            {
                if (inB.Shape[1] == K)
                {
                    transB = true;
                    N = inB.Shape[0];
                }
                else if (inB.Shape[0] == K)
                {
                    transB = false;
                    N = inB.Shape[1];
                }
                else
                {
                    N = inB.Shape[0];
                }
            }
            else
            {
                N = inB.ElementCount;
            }

            float* pA = (float*)(arena.BasePointer + inA.Offset);
            float* pB = (float*)(arena.BasePointer + inB.Offset);
            float* pBias = InputDescs.Count > 2 ? (float*)(arena.BasePointer + InputDescs[2].Offset) : null;
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            FusedKernels.FusedMatMulBiasAct(pA, pB, pBias, pY, M, N, K, Activation, transB);
        }

        private unsafe void ExecuteConv2D(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var inW = InputDescs[1];
            var outY = OutputDescs[0];

            int batch = inX.Shape.Length >= 4 ? inX.Shape[0] : 1;
            int inC = inX.Shape.Length >= 4 ? inX.Shape[1] : 1;
            int inH = inX.Shape.Length >= 4 ? inX.Shape[2] : inX.Shape[0];
            int inWVal = inX.Shape.Length >= 4 ? inX.Shape[3] : inX.Shape[1];

            int outC = inW.Shape.Length >= 4 ? inW.Shape[0] : 1;
            int kH = inW.Shape.Length >= 4 ? inW.Shape[2] : 3;
            int kW = inW.Shape.Length >= 4 ? inW.Shape[3] : 3;

            int sH = Strides.Length > 0 ? Strides[0] : 1;
            int sW = Strides.Length > 1 ? Strides[1] : sH;
            int pH = Pads.Length > 0 ? Pads[0] : 0;
            int pW = Pads.Length > 1 ? Pads[1] : pH;

            float* pX = (float*)(arena.BasePointer + inX.Offset);
            float* pWeight = (float*)(arena.BasePointer + inW.Offset);
            float* pBias = InputDescs.Count > 2 ? (float*)(arena.BasePointer + InputDescs[2].Offset) : null;
            float* pMean = InputDescs.Count > 3 ? (float*)(arena.BasePointer + InputDescs[3].Offset) : null;
            float* pVar = InputDescs.Count > 4 ? (float*)(arena.BasePointer + InputDescs[4].Offset) : null;
            float* pGamma = InputDescs.Count > 5 ? (float*)(arena.BasePointer + InputDescs[5].Offset) : null;
            float* pBeta = InputDescs.Count > 6 ? (float*)(arena.BasePointer + InputDescs[6].Offset) : null;
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            FusedKernels.FusedConv2DBatchNormReLU(
                pX, pWeight, pBias, pMean, pVar, pGamma, pBeta, pY,
                batch, inC, inH, inWVal, outC, kH, kW, sH, sW, pH, pW, Epsilon);
        }

        private unsafe void ExecuteRMSNorm(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];
            int D = inX.Shape.Length > 0 ? inX.Shape[^1] : inX.ElementCount;
            int N = inX.ElementCount / (D > 0 ? D : 1);

            float* pX = (float*)(arena.BasePointer + inX.Offset);
            float* pGamma = InputDescs.Count > 1 ? (float*)(arena.BasePointer + InputDescs[1].Offset) : null;
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            FusedKernels.FusedRMSNorm(pX, pGamma, pY, N, D, Epsilon);
        }

        private unsafe void ExecuteLayerNorm(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];
            int D = inX.Shape.Length > 0 ? inX.Shape[^1] : inX.ElementCount;
            int N = inX.ElementCount / (D > 0 ? D : 1);

            float* pX = (float*)(arena.BasePointer + inX.Offset);
            float* pGamma = InputDescs.Count > 1 ? (float*)(arena.BasePointer + InputDescs[1].Offset) : null;
            float* pBeta = InputDescs.Count > 2 ? (float*)(arena.BasePointer + InputDescs[2].Offset) : null;
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            FusedKernels.FusedLayerNorm(pX, pGamma, pBeta, pY, N, D, Epsilon);
        }

        private unsafe void ExecuteUnaryElementwise(ExecutionMemoryArena arena, ActivationFunction act)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];
            int count = inX.ElementCount;

            float* pX = (float*)(arena.BasePointer + inX.Offset);
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            for (int i = 0; i < count; i++)
            {
                pY[i] = FusedKernels.ApplyActivation(pX[i], act);
            }
        }

        private unsafe void ExecuteBinaryElementwise(ExecutionMemoryArena arena, Func<float, float, float> op)
        {
            var inA = InputDescs[0];
            var inB = InputDescs[1];
            var outY = OutputDescs[0];
            int count = outY.ElementCount;

            float* pA = (float*)(arena.BasePointer + inA.Offset);
            float* pB = (float*)(arena.BasePointer + inB.Offset);
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            if (inA.ElementCount == count && inB.ElementCount == count)
            {
                for (int i = 0; i < count; i++)
                    pY[i] = op(pA[i], pB[i]);
            }
            else if (inB.ElementCount == 1) // Scalar broadcasting
            {
                float scalarB = pB[0];
                for (int i = 0; i < count; i++)
                    pY[i] = op(pA[i], scalarB);
            }
            else if (inA.ElementCount == 1)
            {
                float scalarA = pA[0];
                for (int i = 0; i < count; i++)
                    pY[i] = op(scalarA, pB[i]);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    pY[i] = op(pA[i % inA.ElementCount], pB[i % inB.ElementCount]);
                }
            }
        }

        private unsafe void ExecuteSoftmax(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];
            int D = inX.Shape.Length > 0 ? inX.Shape[^1] : inX.ElementCount;
            int N = inX.ElementCount / (D > 0 ? D : 1);

            float* pX = (float*)(arena.BasePointer + inX.Offset);
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            for (int i = 0; i < N; i++)
            {
                float* rowX = pX + i * D;
                float* rowY = pY + i * D;

                float maxVal = rowX[0];
                for (int j = 1; j < D; j++)
                    if (rowX[j] > maxVal) maxVal = rowX[j];

                float sumExp = 0f;
                for (int j = 0; j < D; j++)
                {
                    float e = MathF.Exp(rowX[j] - maxVal);
                    rowY[j] = e;
                    sumExp += e;
                }

                float invSum = 1.0f / (sumExp > 1e-8f ? sumExp : 1e-8f);
                for (int j = 0; j < D; j++)
                    rowY[j] *= invSum;
            }
        }

        private unsafe void ExecuteAttention(ExecutionMemoryArena arena)
        {
            // Inputs: Q, K, V (and optional Mask)
            var inQ = InputDescs[0];
            var inK = InputDescs[1];
            var inV = InputDescs[2];
            var outY = OutputDescs[0];

            int seqLen = inQ.Shape.Length > 1 ? inQ.Shape[0] : 1;
            int hiddenDim = inQ.Shape.Length > 1 ? inQ.Shape[1] : inQ.ElementCount;

            float* pQ = (float*)(arena.BasePointer + inQ.Offset);
            float* pK = (float*)(arena.BasePointer + inK.Offset);
            float* pV = (float*)(arena.BasePointer + inV.Offset);
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            float scale = 1.0f / MathF.Sqrt(hiddenDim);

            // Q * K^T -> Softmax -> * V
            Span<float> scores = stackalloc float[seqLen * seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                float* qRow = pQ + i * hiddenDim;
                for (int j = 0; j < seqLen; j++)
                {
                    float* kRow = pK + j * hiddenDim;
                    float dot = 0f;
                    for (int d = 0; d < hiddenDim; d++)
                        dot += qRow[d] * kRow[d];
                    scores[i * seqLen + j] = dot * scale;
                }

                // Softmax on row i
                float maxVal = scores[i * seqLen];
                for (int j = 1; j < seqLen; j++)
                    if (scores[i * seqLen + j] > maxVal) maxVal = scores[i * seqLen + j];

                float sumExp = 0f;
                for (int j = 0; j < seqLen; j++)
                {
                    float e = MathF.Exp(scores[i * seqLen + j] - maxVal);
                    scores[i * seqLen + j] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / (sumExp > 1e-8f ? sumExp : 1e-8f);
                for (int j = 0; j < seqLen; j++)
                    scores[i * seqLen + j] *= invSum;

                // Weighted sum with V
                float* yRow = pY + i * hiddenDim;
                for (int d = 0; d < hiddenDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++)
                    {
                        float* vRow = pV + j * hiddenDim;
                        sum += scores[i * seqLen + j] * vRow[d];
                    }
                    yRow[d] = sum;
                }
            }
        }

        private unsafe void ExecuteQuantizedGemm(ExecutionMemoryArena arena)
        {
            var inA = InputDescs[0];
            var inB = InputDescs[1];
            var outY = OutputDescs[0];

            int M = inA.Shape.Length > 1 ? inA.Shape[0] : 1;
            int K = inA.Shape.Length > 1 ? inA.Shape[1] : inA.ElementCount;
            int N = inB.Shape.Length > 1 ? inB.Shape[0] : inB.ElementCount;

            sbyte* pA = (sbyte*)(arena.BasePointer + inA.Offset);
            sbyte* pB = (sbyte*)(arena.BasePointer + inB.Offset);
            float* pBias = InputDescs.Count > 2 ? (float*)(arena.BasePointer + InputDescs[2].Offset) : null;
            float* pY = (float*)(arena.BasePointer + outY.Offset);

            QuantizedMatMulKernels.MatMulInt8Symmetric(pA, pB, pBias, pY, M, N, K, ScaleA, ScaleB);
        }

        private unsafe void ExecuteIdentityOrReshape(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];
            if (inX.Offset != outY.Offset)
            {
                Buffer.MemoryCopy(
                    arena.BasePointer + inX.Offset,
                    arena.BasePointer + outY.Offset,
                    outY.ByteSize,
                    Math.Min(inX.ByteSize, outY.ByteSize));
            }
        }

        private unsafe void ExecuteTranspose(ExecutionMemoryArena arena)
        {
            var inX = InputDescs[0];
            var outY = OutputDescs[0];

            // 2D Transpose
            if (inX.Shape.Length == 2)
            {
                int rows = inX.Shape[0];
                int cols = inX.Shape[1];
                float* pSrc = (float*)(arena.BasePointer + inX.Offset);
                float* pDst = (float*)(arena.BasePointer + outY.Offset);

                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        pDst[c * rows + r] = pSrc[r * cols + c];
                    }
                }
            }
            else
            {
                ExecuteIdentityOrReshape(arena);
            }
        }
    }
}
