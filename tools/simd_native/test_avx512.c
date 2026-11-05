/* test_avx512.c — thu THUC THI mot lenh AVX-512 that su (khong chi CPUID).
 * Neu chay duoc va in dung ket qua -> OS da enable ZMM state, build /arch:AVX512
 * la an toan va co the gap doi be rong FMA so voi AVX2. */
#include <stdio.h>
#include <immintrin.h>

int main(void) {
    float a[16], b[16], c[16];
    for (int i = 0; i < 16; i++) { a[i] = (float)i; b[i] = 2.0f; c[i] = 1.0f; }
    __m512 va = _mm512_loadu_ps(a);
    __m512 vb = _mm512_loadu_ps(b);
    __m512 vc = _mm512_loadu_ps(c);
    __m512 r = _mm512_fmadd_ps(va, vb, vc);
    _mm512_storeu_ps(c, r);
    printf("avx512 fmadd ok: c[0]=%.1f c[15]=%.1f (expect 1.0 / 31.0)\n", c[0], c[15]);
    return (c[15] == 31.0f) ? 0 : 1;
}
