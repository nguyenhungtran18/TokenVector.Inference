/* test_peak.c — TRAN tinh toan AVX2/FMA that cua may, 1 luong.
 * Muc dich: biet tkv 34.5 GFLOP/s (shape 1024x3072x768) la da sat tran chua,
 * hay con headroom so voi torch 78.6 GFLOP/s (cung shape, cung phien).
 *
 * Kernel giu moi thu trong L1: 4 hang x 16 cot, 8 ymm accumulator, 2 load B +
 * 4 broadcast A + 8 FMA moi buoc k - dung cau truc micro_f32 cua
 * tkv_simd_native.c. Lap lai nhieu lan de thoi gian > 0.5 s.
 */
#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define KK 4096
#define ITERS 4000

static __declspec(align(64)) float A[4][KK];
static __declspec(align(64)) float B[KK][16];
static __declspec(align(64)) float Cout[4][16];

static double now_s(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

int main(void) {
    int i, p, j, it;
    float s = 0.0f;
    double t0, dt, flops;

    for (i = 0; i < 4; i++)
        for (p = 0; p < KK; p++) A[i][p] = (float)((i + p) % 7) * 0.1f;
    for (p = 0; p < KK; p++)
        for (j = 0; j < 16; j++) B[p][j] = (float)((p + j) % 5) * 0.2f;

    {
        __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
        __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
        __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
        __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();

        t0 = now_s();
        for (it = 0; it < ITERS; it++) {
            for (p = 0; p < KK; p++) {
                __m256 vb0 = _mm256_load_ps(B[p]);
                __m256 vb1 = _mm256_load_ps(B[p] + 8);
                __m256 va0 = _mm256_set1_ps(A[0][p]);
                __m256 va1 = _mm256_set1_ps(A[1][p]);
                __m256 va2 = _mm256_set1_ps(A[2][p]);
                __m256 va3 = _mm256_set1_ps(A[3][p]);
                c0a = _mm256_fmadd_ps(va0, vb0, c0a);
                c0b = _mm256_fmadd_ps(va0, vb1, c0b);
                c1a = _mm256_fmadd_ps(va1, vb0, c1a);
                c1b = _mm256_fmadd_ps(va1, vb1, c1b);
                c2a = _mm256_fmadd_ps(va2, vb0, c2a);
                c2b = _mm256_fmadd_ps(va2, vb1, c2b);
                c3a = _mm256_fmadd_ps(va3, vb0, c3a);
                c3b = _mm256_fmadd_ps(va3, vb1, c3b);
            }
        }
        dt = now_s() - t0;

        {
            __m256 acc = _mm256_add_ps(
                _mm256_add_ps(c0a, c0b),
                _mm256_add_ps(_mm256_add_ps(c1a, c1b),
                              _mm256_add_ps(_mm256_add_ps(c2a, c2b),
                                            _mm256_add_ps(c3a, c3b))));
            _mm256_store_ps(&Cout[0][0], acc);
        }
    }
    for (j = 0; j < 8; j++) s += Cout[0][j];

    flops = 2.0 * 4 * 16 * (double)KK * (double)ITERS;
    printf("PEAK 1T: %.3f s  %.1f GFLOP/s  (%.2f flop/cycle @2.3GHz)  checksum=%.4f\n",
           dt, flops / dt / 1e9, flops / dt / 2.3e9, s);
    return 0;
}
