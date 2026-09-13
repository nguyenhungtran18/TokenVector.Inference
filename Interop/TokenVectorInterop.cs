using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using TokenVector.Inference.Quantization;
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

namespace TokenVector.Inference.Interop
{
    /// <summary>
    /// Static CIL Interop Bridge for TokenVector Compiler (stdlib/tv/inference).
    /// </summary>
    public static unsafe class TokenVectorInterop
    {
        private static readonly ConcurrentDictionary<IntPtr, InferenceSession> _sessions = new();
        private static long _handleCounter = 1000;

        [UnmanagedCallersOnly(EntryPoint = "tv_inference_create_session")]
        public static IntPtr CreateSession(IntPtr pathUtf8)
        {
            if (pathUtf8 == IntPtr.Zero) return IntPtr.Zero;
            string path = Marshal.PtrToStringUTF8(pathUtf8) ?? string.Empty;

            try
            {
                var session = new InferenceSession(path);
                long handleVal = System.Threading.Interlocked.Increment(ref _handleCounter);
                IntPtr handle = (IntPtr)handleVal;
                _sessions[handle] = session;
                return handle;
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        [UnmanagedCallersOnly(EntryPoint = "tv_inference_run_ptr")]
        public static int RunPointers(IntPtr sessionHandle, float* inPtr, int inLen, float* outPtr, int outLen)
        {
            if (!_sessions.TryGetValue(sessionHandle, out var session))
                return -1;

            try
            {
                var inSpan = new ReadOnlySpan<float>(inPtr, inLen);
                var outSpan = new Span<float>(outPtr, outLen);

                string inName = session.Graph.Inputs.Count > 0 ? session.Graph.Inputs[0] : "input";
                NamedNDArray[] inputs = new NamedNDArray[1]
                {
                    new(inName, new ReadOnlyMemory<float>(inSpan.ToArray()))
                };

                var outArray = new NDArray<float>(new int[] { 1, outLen });
                string outName = session.Graph.Outputs.Count > 0 ? session.Graph.Outputs[0] : "output";
                NamedNDArray[] outputs = new NamedNDArray[1]
                {
                    new(outName, outArray)
                };

                session.Run(inputs, outputs);

                outArray.AsSpan().CopyTo(outSpan);
                return 0;
            }
            catch
            {
                return -2;
            }
        }

        [UnmanagedCallersOnly(EntryPoint = "tv_inference_quantize_int8")]
        public static void QuantizeInt8(float* inPtr, sbyte* outPtr, int len, float scale)
        {
            var src = new ReadOnlySpan<float>(inPtr, len);
            var dst = new Span<sbyte>(outPtr, len);
            QuantizationEngine.QuantizeToInt8Symmetric(src, dst, scale);
        }

        [UnmanagedCallersOnly(EntryPoint = "tv_inference_destroy_session")]
        public static void DestroySession(IntPtr sessionHandle)
        {
            if (_sessions.TryRemove(sessionHandle, out var session))
            {
                session.Dispose();
            }
        }
    }
}
