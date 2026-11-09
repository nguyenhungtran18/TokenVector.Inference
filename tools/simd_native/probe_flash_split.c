/* probe_split.c - serial vs multi-head parallel, fwd and bwd timed separately. */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>

/* fwd: (qkv,qkv_len,ctx,ctx_len,lse,lse_len,n,h,hd,head,inv) = 11 */
typedef int (*fwdlse_t)(const float*, int, float*, int, float*, int,
                        int, int, int, int, double);
/* bwd: (qkv,qkv_len,dctx,dctx_len,ctx_in,ctx_in_len,lse_in,lse_in_len,
         ctxo,ctxo_len,dqo,dqo_len,n,h,hd,head,inv) = 17 */
typedef int (*bwdlse_t)(const float*, int, const float*, int,
                        const float*, int, const float*, int,
                        float*, int, float*, int,
                        int, int, int, int, double);
/* fwd multi: same as fwd but head replaced by nthreads = 11 */
typedef int (*fwdm_t)(const float*, int, float*, int, float*, int,
                      int, int, int, double, int);
/* bwd multi: same as bwd but head replaced by nthreads = 17 */
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

    float* qkv  = (float*)malloc(qkv_len * 4);
    float* dctx = (float*)malloc(dctx_len * 4);
    float* ctx  = (float*)malloc(ctx_len * 4);
    float* lse  = (float*)malloc(lse_len * 4);
    float* dqo  = (float*)malloc(dqo_len * 4);

    srand(12345);
    for (long long i = 0; i < qkv_len; i++)  qkv[i]  = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (long long i = 0; i < dctx_len; i++) dctx[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    HMODULE h = LoadLibraryA("tkv_simd_native.dll");
    if (!h) { printf("load fail\n"); return 1; }
    fwdlse_t fw  = (fwdlse_t)GetProcAddress(h, "FlashAttentionFwdLse");
    bwdlse_t bw  = (bwdlse_t)GetProcAddress(h, "FlashAttentionBwdFusedLse");
    fwdm_t   fwm = (fwdm_t)GetProcAddress(h, "FlashAttentionFwdLseMulti");
    bwdm_t   bwm = (bwdm_t)GetProcAddress(h, "FlashAttentionBwdFusedLseMulti");
    if (!fw || !bw || !fwm || !bwm) { printf("sym fail\n"); return 1; }

    for (int i = 0; i < nh; i++)
        fw(qkv, (int)qkv_len, ctx + (long long)i * n * hd, (int)ctx_len,
           lse + (long long)i * n, n, n, hh, hd, i, inv);

    double bf = 1e30, bb = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms();
        for (int i = 0; i < nh; i++)
            fw(qkv, (int)qkv_len, ctx + (long long)i * n * hd, (int)ctx_len,
               lse + (long long)i * n, n, n, hh, hd, i, inv);
        double t1 = now_ms();
        for (int i = 0; i < nh; i++)
            bw(qkv, (int)qkv_len, dctx, (int)dctx_len,
               ctx + (long long)i * n * hd, (int)ctx_len,
               lse + (long long)i * n, n,
               ctx + (long long)i * n * hd, (int)ctx_len,
               dqo + (long long)i * n * 3 * hd, (int)dqo_len,
               n, hh, hd, i, inv);
        double t2 = now_ms();
        if (t1 - t0 < bf) bf = t1 - t0;
        if (t2 - t1 < bb) bb = t2 - t1;
    }
    printf("SERIAL   fwd=%6.2f  bwd=%6.2f ms/layer\n", bf, bb);

    int tl[5] = {2, 4, 7, 12, 14};
    for (int k = 0; k < 5; k++) {
        int nt = tl[k];
        double mf = 1e30, mb = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = now_ms();
            fwm(qkv, (int)qkv_len, ctx, (int)ctx_len, lse, (int)lse_len,
                n, hh, hd, inv, nt);
            double t1 = now_ms();
            bwm(qkv, (int)qkv_len, dctx, (int)dctx_len,
                ctx, (int)ctx_len, lse, (int)lse_len,
                ctx, (int)ctx_len, dqo, (int)dqo_len,
                n, hh, hd, inv, nt);
            double t2 = now_ms();
            if (t1 - t0 < mf) mf = t1 - t0;
            if (t2 - t1 < mb) mb = t2 - t1;
        }
        printf("MULTI nt=%2d  fwd=%6.2f  bwd=%6.2f ms/layer   bwd speedup %.2fx\n",
               nt, mf, mb, bb / mb);
    }
    return 0;
}
