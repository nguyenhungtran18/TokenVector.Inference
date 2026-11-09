/* probe_mh.c - serial vs multi-head parallel flash bwd/fwd.
 * Verify bit-identical output + measure speedup.
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>

typedef int (*fwdlse_t)(const float*, int, float*, int, float*, int,
                        int, int, int, int, double);
typedef int (*bwdlse_t)(const float*, int, const float*, int,
                        const float*, int, const float*, int,
                        float*, int, float*, int,
                        int, int, int, int, double);
typedef int (*fwdm_t)(const float*, int, float*, int, float*, int,
                      int, int, int, double, int);
typedef int (*bwdm_t)(const float*, int, const float*, int,
                      const float*, int, const float*, int,
                      float*, int, float*, int,
                      int, int, int, double, int);

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    int hh = 768, hd = 64, nh = hh / hd;
    double inv = 1.0 / 8.0;
    long long qkv_len  = (long long)n * 3 * hh;
    long long dctx_len = (long long)n * hh;
    long long ctx_len  = (long long)n * nh * hd;
    long long dqo_len  = (long long)n * nh * 3 * hd;
    long long lse_len  = (long long)n * nh;

    float* qkv   = (float*)malloc(qkv_len * 4);
    float* dctx  = (float*)malloc(dctx_len * 4);
    float* ctx   = (float*)malloc(ctx_len * 4);
    float* lse   = (float*)malloc(lse_len * 4);
    float* dqo   = (float*)malloc(dqo_len * 4);
    float* ctx_m = (float*)malloc(ctx_len * 4);
    float* lse_m = (float*)malloc(lse_len * 4);
    float* dqo_m = (float*)malloc(dqo_len * 4);

    srand(12345);
    for (long long i = 0; i < qkv_len; i++)  qkv[i]  = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (long long i = 0; i < dctx_len; i++) dctx[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    HMODULE h = LoadLibraryA("tkv_simd_native.dll");
    if (!h) { printf("load fail\n"); return 1; }
    fwdlse_t fw  = (fwdlse_t)GetProcAddress(h, "FlashAttentionFwdLse");
    bwdlse_t bw  = (bwdlse_t)GetProcAddress(h, "FlashAttentionBwdFusedLse");
    fwdm_t   fwm = (fwdm_t)GetProcAddress(h, "FlashAttentionFwdLseMulti");
    bwdm_t   bwm = (bwdm_t)GetProcAddress(h, "FlashAttentionBwdFusedLseMulti");
    if (!fw || !bw)   { printf("sym fail base\n");  return 1; }
    if (!fwm || !bwm) { printf("sym fail multi\n"); return 1; }

    /* ---- serial reference ---- */
    for (int i = 0; i < nh; i++) {
        int rc = fw(qkv, (int)qkv_len,
                    ctx + (long long)i * n * hd, (int)ctx_len,
                    lse + (long long)i * n, n,
                    n, hh, hd, i, inv);
        if (rc) { printf("fwd rc=%d\n", rc); return 2; }
    }
    for (int i = 0; i < nh; i++) {
        int rc = bw(qkv, (int)qkv_len,
                    dctx, (int)dctx_len,
                    ctx + (long long)i * n * hd, (int)ctx_len,
                    lse + (long long)i * n, n,
                    ctx + (long long)i * n * hd, (int)ctx_len,
                    dqo + (long long)i * n * 3 * hd, (int)dqo_len,
                    n, hh, hd, i, inv);
        if (rc) { printf("bwd rc=%d\n", rc); return 3; }
    }

    /* ---- multi-head ---- */
    int rc = fwm(qkv, (int)qkv_len, ctx_m, (int)ctx_len, lse_m, (int)lse_len,
                 n, hh, hd, inv, nh);
    printf("fwd_multi rc=%d\n", rc);
    rc = bwm(qkv, (int)qkv_len, dctx, (int)dctx_len,
             ctx_m, (int)ctx_len, lse_m, (int)lse_len,
             ctx_m, (int)ctx_len, dqo_m, (int)dqo_len,
             n, hh, hd, inv, nh);
    printf("bwd_multi rc=%d\n", rc);

    double md = 0.0;
    for (long long i = 0; i < dqo_len; i++) {
        double d = fabs((double)dqo[i] - (double)dqo_m[i]);
        if (d > md) md = d;
    }
    printf("dqo serial-vs-multi max_abs = %.3e\n", md);
    md = 0.0;
    for (long long i = 0; i < ctx_len; i++) {
        double d = fabs((double)ctx[i] - (double)ctx_m[i]);
        if (d > md) md = d;
    }
    printf("ctx serial-vs-multi max_abs = %.3e\n", md);

    /* ---- timing ---- */
    double bests = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms();
        for (int i = 0; i < nh; i++)
            fw(qkv, (int)qkv_len, ctx + (long long)i * n * hd, (int)ctx_len,
               lse + (long long)i * n, n, n, hh, hd, i, inv);
        for (int i = 0; i < nh; i++)
            bw(qkv, (int)qkv_len, dctx, (int)dctx_len,
               ctx + (long long)i * n * hd, (int)ctx_len,
               lse + (long long)i * n, n,
               ctx + (long long)i * n * hd, (int)ctx_len,
               dqo + (long long)i * n * 3 * hd, (int)dqo_len,
               n, hh, hd, i, inv);
        double t1 = now_ms();
        if (t1 - t0 < bests) bests = t1 - t0;
    }
    printf("\nSERIAL  fwd+bwd layer: %7.2f ms\n", bests);

    int tlist[5] = {2, 4, 7, 12, 14};
    for (int k = 0; k < 5; k++) {
        int nt = tlist[k];
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = now_ms();
            fwm(qkv, (int)qkv_len, ctx_m, (int)ctx_len, lse_m, (int)lse_len,
                n, hh, hd, inv, nt);
            bwm(qkv, (int)qkv_len, dctx, (int)dctx_len,
                ctx_m, (int)ctx_len, lse_m, (int)lse_len,
                ctx_m, (int)ctx_len, dqo_m, (int)dqo_len,
                n, hh, hd, inv, nt);
            double t1 = now_ms();
            if (t1 - t0 < best) best = t1 - t0;
        }
        printf("MULTI nt=%2d  fwd+bwd layer: %7.2f ms   speedup %.2fx\n",
               nt, best, bests / best);
    }
    return 0;
}
