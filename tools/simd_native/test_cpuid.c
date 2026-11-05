/* test_cpuid.c — kiem tra CPU co AVX2 / AVX-512F that khong (khong dung /arch).
 * Chay: cl /nologo test_cpuid.c && test_cpuid.exe
 * Ly do: Win32_Processor bao "Xeon E5-2695 v3" (Haswell, khong co AVX-512) nhung
 * torch dat 26 FLOP/cycle/core — vuot tran AVX2. Can biet that de chon arch. */
#include <stdio.h>

static void cpuid(int leaf, int sub, int out[4]) {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, sub);
#else
    __asm__ __volatile__("cpuid" : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                         : "a"(leaf), "c"(sub));
#endif
}

int main(void) {
    int r[4];
    cpuid(0, 0, r);
    char v[13];
    *(int*)&v[0] = r[1]; *(int*)&v[4] = r[3]; *(int*)&v[8] = r[2]; v[12] = 0;
    printf("vendor: %s\n", v);

    cpuid(1, 0, r);
    int osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    int fma = (r[2] >> 12) & 1;
    printf("OSXSAVE=%d AVX=%d FMA=%d\n", osxsave, avx, fma);

    /* XCR0: can bit1 (SSE) + bit2 (AVX) + bit5,6,7 (opmask/zmm) cho AVX-512 */
    unsigned long long xcr0 = 0;
#if defined(_MSC_VER)
    xcr0 = _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ __volatile__(".byte 0x0f,0x01,0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = ((unsigned long long)edx << 32) | eax;
#endif
    printf("XCR0=0x%llx\n", xcr0);

    cpuid(7, 0, r);
    int avx2 = (r[1] >> 5) & 1;
    int avx512f = (r[1] >> 16) & 1;
    int avx512dq = (r[1] >> 17) & 1;
    int avx512bw = (r[1] >> 30) & 1;
    int avx512vl = (r[1] >> 31) & 1;
    printf("AVX2=%d AVX512F=%d DQ=%d BW=%d VL=%d\n", avx2, avx512f, avx512dq, avx512bw, avx512vl);

    int usable = avx512f && ((xcr0 & 0xE6) == 0xE6);
    printf("AVX512 usable = %s\n", usable ? "YES" : "NO");
    return 0;
}
