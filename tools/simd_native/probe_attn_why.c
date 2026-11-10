/* probe_attn_why.c - vi sao native bwd (119 ms/layer) cham hon ATen (65)?
 *
 * Tach 3 tang cua duong production `attention_bwd_merge_multi`:
 *   T1) FlashAttentionFwdLseMulti        - recompute fwd (ctx + lse)
 *   T2) FlashAttentionBwdFusedLseMulti   - bwd kernel thuan (ctx/lse da co)
 *   T3) FlashAttentionBwdMergeMulti      - bwd kernel + scatter thang
 *   P ) AttentionBwdMergeMulti           - DUONG PRODUCTION = T1 + T3
 *
 * Do them: CreateThread cost, malloc cost, scaling theo nthreads.
 *
 * Build (tu tools/simd_native):
 *   powershell -ExecutionPolicy Bypass -File build_probe_attn_why.ps1
 * Chay: copy out\tkv_simd_native.dll vao cung thu muc roi ./probe_attn_why.exe 1024
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>

typedef int (*fwdm_t)(const float*, int, float*, int, float*, int,
                      int, int, int, double, int);
typedef int (*bwdm_t)(const float*, int, const float*, int,
                      const float*, int, const float*, int,
                      float*, int, float*, int,
                      int, int, int, double, int);
typedef int (*mergem_t)(const float*, int, const float*, int,
                        const float*, int, const float*, int,
                        float*, int, float*, int,
                        int, int, int, double, int);
typedef int (*prod_t)(const float*, int, const float*, int,
                      float*, int, float*, int,
                      int, int, int, double, int);

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double best_of(int reps, void (*body)(void*), void* arg) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        body(arg);
        double t1 = now_ms();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

/* ---- globals cho body ---- */
static fwdm_t   g_fwd;
static bwdm_t   g_bwd;
static mergem_t g_merge;
static prod_t   g_prod;
static int g_n, g_h, g_hd, g_nh, g_nt;
static double g_inv;
static float *qkv, *dctx, *ctxf, *lsef, *ctxo, *dqo, *ctxT, *dqkv;
static long long qkv_len, dctx_len, ctx_all, lse_all, dqo_all, dqkv_len;

static void body_fwd(void* p) {
    (void)p;
    g_fwd(qkv, (int)qkv_len, ctxf, (int)ctx_all, lsef, (int)lse_all,
          g_n, g_h, g_hd, g_inv, g_nt);
}
static void body_bwd(void* p) {
    (void)p;
    g_bwd(qkv, (int)qkv_len, dctx, (int)dctx_len,
          ctxf, (int)ctx_all, lsef, (int)lse_all,
          ctxo, (int)ctx_all, dqo, (int)dqo_all,
          g_n, g_h, g_hd, g_inv, g_nt);
}
static void body_merge(void* p) {
    (void)p;
    g_merge(qkv, (int)qkv_len, dctx, (int)dctx_len,
            ctxf, (int)ctx_all, lsef, (int)lse_all,
            ctxT, (int)ctx_all, dqkv, (int)dqkv_len,
            g_n, g_h, g_hd, g_inv, g_nt);
}
static void body_prod(void* p) {
    (void)p;
    g_prod(qkv, (int)qkv_len, dctx, (int)dctx_len,
           ctxT, (int)ctx_all, dqkv, (int)dqkv_len,
           g_n, g_h, g_hd, g_inv, g_nt);
}

/* ---- CreateThread cost: tao + join N thread rong ---- */
static DWORD WINAPI noop(LPVOID p) { (void)p; return 0; }
static void body_threads(void* p) {
    int nt = *(int*)p;
    HANDLE th[64];
    for (int i = 0; i < nt; i++) th[i] = CreateThread(NULL, 0, noop, NULL, 0, NULL);
    WaitForMultipleObjects(nt, th, TRUE, INFINITE);
    for (int i = 0; i < nt; i++) CloseHandle(th[i]);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    int h = 768, hd = 64, nh = h / hd;
    double inv = 1.0 / 8.0;
    int reps = 5;

    qkv_len  = (long long)n * 3 * h;
    dctx_len = (long long)n * h;
    ctx_all  = (long long)n * nh * hd;   /* == n*h */
    lse_all  = (long long)n * nh;
    dqo_all  = (long long)n * nh * 3 * hd; /* == n*3*h */
    dqkv_len = dqo_all;

    qkv  = (float*)malloc(qkv_len * 4);
    dctx = (float*)malloc(dctx_len * 4);
    ctxf = (float*)malloc(ctx_all * 4);
    lsef = (float*)malloc(lse_all * 4);
    ctxo = (float*)malloc(ctx_all * 4);
    dqo  = (float*)malloc(dqo_all * 4);
    ctxT = (float*)malloc(ctx_all * 4);
    dqkv = (float*)malloc(dqo_all * 4);

    srand(12345);
    for (long long i = 0; i < qkv_len; i++)  qkv[i]  = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (long long i = 0; i < dctx_len; i++) dctx[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    HMODULE hm = LoadLibraryA("tkv_simd_native.dll");
    if (!hm) { printf("load fail\n"); return 1; }
    g_fwd   = (fwdm_t)  GetProcAddress(hm, "FlashAttentionFwdLseMulti");
    g_bwd   = (bwdm_t)  GetProcAddress(hm, "FlashAttentionBwdFusedLseMulti");
    g_merge = (mergem_t)GetProcAddress(hm, "FlashAttentionBwdMergeMulti");
    g_prod  = (prod_t)  GetProcAddress(hm, "AttentionBwdMergeMulti");
    if (!g_fwd || !g_bwd || !g_merge || !g_prod) {
        printf("sym fail: fwd=%p bwd=%p merge=%p prod=%p\n",
               (void*)g_fwd, (void*)g_bwd, (void*)g_merge, (void*)g_prod);
        return 1;
    }

    g_n = n; g_h = h; g_hd = hd; g_nh = nh; g_inv = inv;

    printf("=== n=%d h=%d hd=%d nh=%d, best of %d, ms/layer ===\n\n", n, h, hd, nh, reps);

    /* --- A: scaling theo nthreads cho tung tang --- */
    int tl[5] = {1, 2, 4, 8, 12};
    printf("%-6s | %8s | %8s | %8s | %8s\n", "nt", "fwd(T1)", "bwd(T2)", "merge(T3)", "prod(P)");
    printf("-------|----------|----------|----------|---------\n");
    double prod12 = 0;
    for (int k = 0; k < 5; k++) {
        g_nt = tl[k];
        double a = best_of(reps, body_fwd,   NULL);
        double b = best_of(reps, body_bwd,   NULL);
        double c = best_of(reps, body_merge, NULL);
        double d = best_of(reps, body_prod,  NULL);
        if (tl[k] == 12) prod12 = d;
        printf("%-6d | %8.2f | %8.2f | %8.2f | %8.2f\n", tl[k], a, b, c, d);
    }

    /* --- B: CreateThread cost --- */
    printf("\n=== CreateThread + join, N thread rong ===\n");
    for (int k = 0; k < 5; k++) {
        int nt = tl[k];
        double t = best_of(20, body_threads, &nt);
        printf("  nt=%2d : %7.3f ms  (%.1f us/thread)\n", nt, t, t * 1000.0 / nt);
    }

    /* --- C: phan ra duong production tai nt=12 --- */
    g_nt = 12;
    double t1 = best_of(reps, body_fwd,   NULL);
    double t2 = best_of(reps, body_bwd,   NULL);
    double t3 = best_of(reps, body_merge, NULL);
    double tp = best_of(reps, body_prod,  NULL);
    printf("\n=== Phan ra AttentionBwdMergeMulti @nt=12 ===\n");
    printf("  T1 FlashAttentionFwdLseMulti (recompute fwd) : %8.2f ms\n", t1);
    printf("  T2 FlashAttentionBwdFusedLseMulti (bwd thuan) : %8.2f ms\n", t2);
    printf("  T3 FlashAttentionBwdMergeMulti (bwd+scatter)  : %8.2f ms\n", t3);
    printf("  P  AttentionBwdMergeMulti (PRODUCTION)        : %8.2f ms\n", tp);
    printf("  ---\n");
    printf("  scatter = T3 - T2                             : %8.2f ms\n", t3 - t2);
    printf("  glue    = P - T1 - T3                         : %8.2f ms\n", tp - t1 - t3);
    printf("  fwd-recompute chiem                          : %8.1f %% cua P\n", 100.0 * t1 / tp);
    printf("  neu BO fwd-recompute (dung ctx/lse co san)   : %8.2f ms  (%.2fx nhanh hon)\n",
           t3, tp / t3);

    /* --- D: verify bit-identical giua T3 va P --- */
    g_nt = 12;
    for (long long i = 0; i < ctx_all; i++) ctxT[i] = -12345.0f;
    for (long long i = 0; i < dqkv_len; i++) dqkv[i] = -12345.0f;
    int rc1 = g_prod(qkv, (int)qkv_len, dctx, (int)dctx_len,
                     ctxT, (int)ctx_all, dqkv, (int)dqkv_len,
                     n, h, hd, inv, 12);
    float* ctxT2 = (float*)malloc(ctx_all * 4);
    float* dqkv2 = (float*)malloc(dqo_all * 4);
    for (long long i = 0; i < ctx_all; i++) ctxT2[i] = -12345.0f;
    for (long long i = 0; i < dqo_all; i++) dqkv2[i] = -12345.0f;
    int rc2 = g_merge(qkv, (int)qkv_len, dctx, (int)dctx_len,
                      ctxf, (int)ctx_all, lsef, (int)lse_all,
                      ctxT2, (int)ctx_all, dqkv2, (int)dqo_all,
                      n, h, hd, inv, 12);
    double md = 0; long long nbad = 0;
    for (long long i = 0; i < ctx_all; i++) {
        double d = fabs((double)ctxT[i] - (double)ctxT2[i]);
        if (d > md) md = d; if (d != 0.0) nbad++;
    }
    for (long long i = 0; i < dqo_all; i++) {
        double d = fabs((double)dqkv[i] - (double)dqkv2[i]);
        if (d > md) md = d; if (d != 0.0) nbad++;
    }
    printf("\n=== Verify P vs T3 (ctxT + dqkv) ===\n");
    printf("  rc prod=%d merge=%d  max_abs=%.3e  lech=%lld\n", rc1, rc2, md, nbad);

    return 0;
}
