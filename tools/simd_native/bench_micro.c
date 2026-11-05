/* bench_micro.c — so sanh micro-kernel GEMM tren dung shape BWD mlp (seq1024).
 *
 * Van de: micro_f32 hien tai la 4x16 = 8 ymm accumulator. FMA tren Haswell co
 * latency 5 chu ky, ma moi accumulator chi duoc cap nhat 1 lan moi buoc k:
 * 8 chuoi / 5 chu ky = 1.6 FMA/chu ky -> tran ~80% dinh, va thuc do chi ~44
 * GFLOP/s tren dinh may ~75.5 (OpenBLAS va torch cung dat 75.7/78.6).
 *
 * Bien the 5x16 (10 acc) va 6x16 (12 acc) tang so chuoi doc lap -> che lat
 * latency. CA HAI giu nguyen thu tu tich luy p tang dan cho tung phan tu
 * (mot accumulator / mot phan tu, FMA lien tiep theo p) nen BIT-EXACT voi V0 —
 * kiem tra bang memcmp, khong suy luan.
 *
 * Build: xem run_bench_micro.ps1 (/O2 /arch:AVX2 /fp:fast, giong build.ps1)
 * Chay:  bench_micro.exe [m] [n] [k] [kc]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

#define GEMM_KC_DEF 64
#define GEMM_MC 64
#define GEMM_NC 64
#define GEMM_BPL (GEMM_NC + 4)

static double now_s(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

/* ---------------- V0: ban sao nguyen van micro_f32 (4x16) ---------------- */
static void pack_b_f32(const float* b, float* bp, int n, int pc, int kc, int jc, int nc, int bpl) {
    for (int p = 0; p < kc; p++) {
        const float* src = b + (long long)(pc + p) * n + jc;
        float* dst = bp + (long long)p * bpl;
        for (int j = 0; j < nc; j++) dst[j] = src[j];
    }
}

static void micro_4x16(const float* a, int lda, const float* bp, float* c, int n,
                       int mc, int nc, int kc, int bpl) {
    int i = 0;
    for (; i + 4 <= mc; i += 4) {
        const float* ai = a + (long long)i * lda;
        float* ci = c + (long long)i * n;
        int j = 0;
        for (; j + 16 <= nc; j += 16) {
            __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
            __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
            __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
            __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
            for (int p = 0; p < kc; p++) {
                const float* brow = bp + (long long)p * bpl + j;
                __m256 vb0 = _mm256_loadu_ps(brow);
                __m256 vb1 = _mm256_loadu_ps(brow + 8);
                __m256 va0 = _mm256_set1_ps(ai[p]);
                __m256 va1 = _mm256_set1_ps(ai[lda + p]);
                __m256 va2 = _mm256_set1_ps(ai[2 * lda + p]);
                __m256 va3 = _mm256_set1_ps(ai[3 * lda + p]);
                c0a = _mm256_fmadd_ps(va0, vb0, c0a);
                c0b = _mm256_fmadd_ps(va0, vb1, c0b);
                c1a = _mm256_fmadd_ps(va1, vb0, c1a);
                c1b = _mm256_fmadd_ps(va1, vb1, c1b);
                c2a = _mm256_fmadd_ps(va2, vb0, c2a);
                c2b = _mm256_fmadd_ps(va2, vb1, c2b);
                c3a = _mm256_fmadd_ps(va3, vb0, c3a);
                c3b = _mm256_fmadd_ps(va3, vb1, c3b);
            }
            float* cj = ci + j;
            _mm256_storeu_ps(cj, _mm256_add_ps(_mm256_loadu_ps(cj), c0a));
            _mm256_storeu_ps(cj + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 8), c0b));
            _mm256_storeu_ps(cj + n, _mm256_add_ps(_mm256_loadu_ps(cj + n), c1a));
            _mm256_storeu_ps(cj + n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + n + 8), c1b));
            _mm256_storeu_ps(cj + 2 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n), c2a));
            _mm256_storeu_ps(cj + 2 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n + 8), c2b));
            _mm256_storeu_ps(cj + 3 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n), c3a));
            _mm256_storeu_ps(cj + 3 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n + 8), c3b));
        }
        for (; j < nc; j++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int p = 0; p < kc; p++) {
                float bv = bp[(long long)p * bpl + j];
                s0 += ai[p] * bv;
                s1 += ai[lda + p] * bv;
                s2 += ai[2 * lda + p] * bv;
                s3 += ai[3 * lda + p] * bv;
            }
            ci[j] += s0; ci[n + j] += s1; ci[2 * n + j] += s2; ci[3 * n + j] += s3;
        }
    }
    for (; i < mc; i++) {
        const float* ai = a + (long long)i * lda;
        float* ci = c + (long long)i * n;
        for (int j = 0; j < nc; j++) {
            float s = 0.0f;
            for (int p = 0; p < kc; p++) s += ai[p] * bp[(long long)p * bpl + j];
            ci[j] += s;
        }
    }
}

/* ---------------- V5: 5x16 (10 accumulator) ---------------- */
static void micro_5x16(const float* a, int lda, const float* bp, float* c, int n,
                       int mc, int nc, int kc, int bpl) {
    int i = 0;
    for (; i + 5 <= mc; i += 5) {
        const float* ai = a + (long long)i * lda;
        float* ci = c + (long long)i * n;
        int j = 0;
        for (; j + 16 <= nc; j += 16) {
            __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
            __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
            __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
            __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
            __m256 c4a = _mm256_setzero_ps(), c4b = _mm256_setzero_ps();
            for (int p = 0; p < kc; p++) {
                const float* brow = bp + (long long)p * bpl + j;
                __m256 vb0 = _mm256_loadu_ps(brow);
                __m256 vb1 = _mm256_loadu_ps(brow + 8);
                __m256 va0 = _mm256_set1_ps(ai[p]);
                __m256 va1 = _mm256_set1_ps(ai[lda + p]);
                __m256 va2 = _mm256_set1_ps(ai[2 * lda + p]);
                __m256 va3 = _mm256_set1_ps(ai[3 * lda + p]);
                __m256 va4 = _mm256_set1_ps(ai[4 * lda + p]);
                c0a = _mm256_fmadd_ps(va0, vb0, c0a);
                c0b = _mm256_fmadd_ps(va0, vb1, c0b);
                c1a = _mm256_fmadd_ps(va1, vb0, c1a);
                c1b = _mm256_fmadd_ps(va1, vb1, c1b);
                c2a = _mm256_fmadd_ps(va2, vb0, c2a);
                c2b = _mm256_fmadd_ps(va2, vb1, c2b);
                c3a = _mm256_fmadd_ps(va3, vb0, c3a);
                c3b = _mm256_fmadd_ps(va3, vb1, c3b);
                c4a = _mm256_fmadd_ps(va4, vb0, c4a);
                c4b = _mm256_fmadd_ps(va4, vb1, c4b);
            }
            float* cj = ci + j;
            _mm256_storeu_ps(cj, _mm256_add_ps(_mm256_loadu_ps(cj), c0a));
            _mm256_storeu_ps(cj + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 8), c0b));
            _mm256_storeu_ps(cj + n, _mm256_add_ps(_mm256_loadu_ps(cj + n), c1a));
            _mm256_storeu_ps(cj + n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + n + 8), c1b));
            _mm256_storeu_ps(cj + 2 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n), c2a));
            _mm256_storeu_ps(cj + 2 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n + 8), c2b));
            _mm256_storeu_ps(cj + 3 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n), c3a));
            _mm256_storeu_ps(cj + 3 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n + 8), c3b));
            _mm256_storeu_ps(cj + 4 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 4 * n), c4a));
            _mm256_storeu_ps(cj + 4 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 4 * n + 8), c4b));
        }
        if (j < nc) micro_4x16(ai, lda, bp, ci, n, 5, nc - j, kc, bpl);
    }
    if (i < mc) micro_4x16(a + (long long)i * lda, lda, bp, c + (long long)i * n, n, mc - i, nc, kc, bpl);
}

/* ---------------- V6: 6x16 (12 accumulator) ---------------- */
static void micro_6x16(const float* a, int lda, const float* bp, float* c, int n,
                       int mc, int nc, int kc, int bpl) {
    int i = 0;
    for (; i + 6 <= mc; i += 6) {
        const float* ai = a + (long long)i * lda;
        float* ci = c + (long long)i * n;
        int j = 0;
        for (; j + 16 <= nc; j += 16) {
            __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
            __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
            __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
            __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
            __m256 c4a = _mm256_setzero_ps(), c4b = _mm256_setzero_ps();
            __m256 c5a = _mm256_setzero_ps(), c5b = _mm256_setzero_ps();
            for (int p = 0; p < kc; p++) {
                const float* brow = bp + (long long)p * bpl + j;
                __m256 vb0 = _mm256_loadu_ps(brow);
                __m256 vb1 = _mm256_loadu_ps(brow + 8);
                __m256 va0 = _mm256_set1_ps(ai[p]);
                __m256 va1 = _mm256_set1_ps(ai[lda + p]);
                c0a = _mm256_fmadd_ps(va0, vb0, c0a);
                c0b = _mm256_fmadd_ps(va0, vb1, c0b);
                c1a = _mm256_fmadd_ps(va1, vb0, c1a);
                c1b = _mm256_fmadd_ps(va1, vb1, c1b);
                __m256 va2 = _mm256_set1_ps(ai[2 * lda + p]);
                __m256 va3 = _mm256_set1_ps(ai[3 * lda + p]);
                c2a = _mm256_fmadd_ps(va2, vb0, c2a);
                c2b = _mm256_fmadd_ps(va2, vb1, c2b);
                c3a = _mm256_fmadd_ps(va3, vb0, c3a);
                c3b = _mm256_fmadd_ps(va3, vb1, c3b);
                __m256 va4 = _mm256_set1_ps(ai[4 * lda + p]);
                __m256 va5 = _mm256_set1_ps(ai[5 * lda + p]);
                c4a = _mm256_fmadd_ps(va4, vb0, c4a);
                c4b = _mm256_fmadd_ps(va4, vb1, c4b);
                c5a = _mm256_fmadd_ps(va5, vb0, c5a);
                c5b = _mm256_fmadd_ps(va5, vb1, c5b);
            }
            float* cj = ci + j;
            _mm256_storeu_ps(cj, _mm256_add_ps(_mm256_loadu_ps(cj), c0a));
            _mm256_storeu_ps(cj + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 8), c0b));
            _mm256_storeu_ps(cj + n, _mm256_add_ps(_mm256_loadu_ps(cj + n), c1a));
            _mm256_storeu_ps(cj + n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + n + 8), c1b));
            _mm256_storeu_ps(cj + 2 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n), c2a));
            _mm256_storeu_ps(cj + 2 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n + 8), c2b));
            _mm256_storeu_ps(cj + 3 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n), c3a));
            _mm256_storeu_ps(cj + 3 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n + 8), c3b));
            _mm256_storeu_ps(cj + 4 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 4 * n), c4a));
            _mm256_storeu_ps(cj + 4 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 4 * n + 8), c4b));
            _mm256_storeu_ps(cj + 5 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 5 * n), c5a));
            _mm256_storeu_ps(cj + 5 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 5 * n + 8), c5b));
        }
        if (j < nc) micro_4x16(ai, lda, bp, ci, n, 6, nc - j, kc, bpl);
    }
    if (i < mc) micro_4x16(a + (long long)i * lda, lda, bp, c + (long long)i * n, n, mc - i, nc, kc, bpl);
}

/* ---------------- V7: 4x16 nhung chi dung phep tang con tro ----------------
 * Ly do (do tu dumpbin): ban 4x16 o tren bi MSVC sinh movsxd/imul CHO TUNG buoc p
 * de tinh dia chi -> tranh port 5 voi vbroadcastss => broadcast bi ALU gianh mat.
 * O day moi con tro tang 1 lan/buoc, khong nhan 64-bit.
 * Thu tu FMA cho tung phan tu (p tang dan) KHONG doi -> bit-exact. */
static void micro_4x16p(const float* a, int lda, const float* bp, float* c, int n,
                        int mc, int nc, int kc, int bpl) {
    int i = 0;
    for (; i + 4 <= mc; i += 4) {
        const float* ai = a + (long long)i * lda;
        float* ci = c + (long long)i * n;
        int j = 0;
        for (; j + 16 <= nc; j += 16) {
            const float* a0 = ai, *a1 = ai + lda, *a2 = ai + 2 * lda, *a3 = ai + 3 * lda;
            const float* bq = bp + j;
            __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
            __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
            __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
            __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
            for (int p = 0; p < kc; p++) {
                __m256 vb0 = _mm256_loadu_ps(bq);
                __m256 vb1 = _mm256_loadu_ps(bq + 8);
                __m256 va0 = _mm256_broadcast_ss(a0);
                __m256 va1 = _mm256_broadcast_ss(a1);
                __m256 va2 = _mm256_broadcast_ss(a2);
                __m256 va3 = _mm256_broadcast_ss(a3);
                c0a = _mm256_fmadd_ps(va0, vb0, c0a);
                c0b = _mm256_fmadd_ps(va0, vb1, c0b);
                c1a = _mm256_fmadd_ps(va1, vb0, c1a);
                c1b = _mm256_fmadd_ps(va1, vb1, c1b);
                c2a = _mm256_fmadd_ps(va2, vb0, c2a);
                c2b = _mm256_fmadd_ps(va2, vb1, c2b);
                c3a = _mm256_fmadd_ps(va3, vb0, c3a);
                c3b = _mm256_fmadd_ps(va3, vb1, c3b);
                bq += bpl; a0++; a1++; a2++; a3++;
            }
            float* cj = ci + j;
            _mm256_storeu_ps(cj, _mm256_add_ps(_mm256_loadu_ps(cj), c0a));
            _mm256_storeu_ps(cj + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 8), c0b));
            _mm256_storeu_ps(cj + n, _mm256_add_ps(_mm256_loadu_ps(cj + n), c1a));
            _mm256_storeu_ps(cj + n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + n + 8), c1b));
            _mm256_storeu_ps(cj + 2 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n), c2a));
            _mm256_storeu_ps(cj + 2 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 2 * n + 8), c2b));
            _mm256_storeu_ps(cj + 3 * n, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n), c3a));
            _mm256_storeu_ps(cj + 3 * n + 8, _mm256_add_ps(_mm256_loadu_ps(cj + 3 * n + 8), c3b));
        }
        for (; j < nc; j++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int p = 0; p < kc; p++) {
                float bv = bp[(long long)p * bpl + j];
                s0 += ai[p] * bv;
                s1 += ai[lda + p] * bv;
                s2 += ai[2 * lda + p] * bv;
                s3 += ai[3 * lda + p] * bv;
            }
            ci[j] += s0; ci[n + j] += s1; ci[2 * n + j] += s2; ci[3 * n + j] += s3;
        }
    }
    if (i < mc) micro_4x16(a, lda, bp, c, n, mc, nc, kc, bpl);
}

typedef void (*micro_fn)(const float*, int, const float*, float*, int, int, int, int, int);

/* ---------------- driver: ban sao gemm_rect_f32 cua production ---------------- */
static void gemm_rect(micro_fn fn, const float* a, int lda, const float* b, int ldb,
                      float* o, int ldc, int m, int n, int k, int kc_g, int mc_g, int nc_g) {
    static __declspec(thread) __declspec(align(32)) float bp[1024 * (GEMM_BPL + 8)];
    int bpl = GEMM_BPL;
    for (int ii = 0; ii < m; ii++)
        memset(o + (long long)ii * ldc, 0, (size_t)n * sizeof(float));
    for (int jc = 0; jc < n; jc += nc_g) {
        int nc = n - jc; if (nc > nc_g) nc = nc_g;
        for (int pc = 0; pc < k; pc += kc_g) {
            int kc = k - pc; if (kc > kc_g) kc = kc_g;
            pack_b_f32(b, bp, ldb, pc, kc, jc, nc, bpl);
            for (int ic = 0; ic < m; ic += mc_g) {
                int mc = m - ic; if (mc > mc_g) mc = mc_g;
                fn(a + (long long)ic * lda + pc, lda, bp, o + (long long)ic * ldc + jc,
                   ldc, mc, nc, kc, bpl);
            }
        }
    }
}

static void fill(float* p, long long cnt, int seed) {
    for (long long i = 0; i < cnt; i++) p[i] = (float)((i * 7 + seed) % 13) * 0.01f - 0.06f;
}

int main(int argc, char** argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 1024;
    int n = (argc > 2) ? atoi(argv[2]) : 3072;
    int k = (argc > 3) ? atoi(argv[3]) : 768;
    int kc_g = (argc > 4) ? atoi(argv[4]) : GEMM_KC_DEF;

    float* A = (float*)malloc((size_t)m * k * sizeof(float));
    float* B = (float*)malloc((size_t)k * n * sizeof(float));
    float* C0 = (float*)malloc((size_t)m * n * sizeof(float));
    float* Cx = (float*)malloc((size_t)m * n * sizeof(float));
    fill(A, (long long)m * k, 3);
    fill(B, (long long)k * n, 5);

    const char* names[4] = {"V0 4x16 (production)", "V5 5x16", "V6 6x16", "V7 4x16 ptr-inc"};
    micro_fn fns[4] = {micro_4x16, micro_5x16, micro_6x16, micro_4x16p};

    if (argc > 5 && argv[5][0] == 'P') {
        /* PEAK mode: goi thang micro-kernel, mc=4 nc=16, du lieu trong L1, kc lon.
         * Tach hoan toan khoi pack_b_f32 va read-modify-write C. */
        int kcp = 2048, iters = 4000;
        static __declspec(align(32)) float Ap[4 * 4096], Bp[4096 * 24], Cp[4 * 16];
        fill(Ap, 4 * 4096, 3);
        fill(Bp, 4096 * 24, 5);
        for (int v = 0; v < 4; v++) {
            memset(Cp, 0, sizeof Cp);
            double best = 1e18;
            for (int r = 0; r < 5; r++) {
                double t0 = now_s();
                for (int it = 0; it < iters; it++)
                    fns[v](Ap, 4096, Bp, Cp, 16, 4, 16, kcp, 24);
                double dt = now_s() - t0;
                if (dt < best) best = dt;
            }
            double fl = 2.0 * 4 * 16 * (double)kcp * iters;
            printf("PEAK %-22s %8.3f s  %6.1f GFLOP/s\n", names[v], best, fl / best / 1e9);
        }
        free(A); free(B); free(C0); free(Cx);
        return 0;
    }

    printf("BM m=%d n=%d k=%d kc=%d mc=%d nc=%d\n", m, n, k, kc_g, GEMM_MC, GEMM_NC);
    gemm_rect(micro_4x16, A, k, B, n, C0, n, m, n, k, kc_g, GEMM_MC, GEMM_NC);

    for (int v = 0; v < 4; v++) {
        double best = 1e18;
        gemm_rect(fns[v], A, k, B, n, Cx, n, m, n, k, kc_g, GEMM_MC, GEMM_NC);
        int same = (memcmp(C0, Cx, (size_t)m * n * sizeof(float)) == 0);
        for (int r = 0; r < 5; r++) {
            memset(Cx, 0, (size_t)m * n * sizeof(float));
            double t0 = now_s();
            gemm_rect(fns[v], A, k, B, n, Cx, n, m, n, k, kc_g, GEMM_MC, GEMM_NC);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        int same2 = (memcmp(C0, Cx, (size_t)m * n * sizeof(float)) == 0);
        double fl = 2.0 * m * n * k;
        printf("BM %-22s %8.1f ms  %6.1f GFLOP/s  bitexact=%s\n",
               names[v], best * 1000.0, fl / best / 1e9, (same && same2) ? "YES" : "NO");
    }
    free(A); free(B); free(C0); free(Cx);
    return 0;
}
