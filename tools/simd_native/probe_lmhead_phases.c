/* probe_lmhead_phases.c - lmhead_bwd_fused tach 5 pha
 *
 * Boi canh (BENCHMARKS.md §8): lmhead_bwd_fused_list = 515 ms/step, la manh
 * LON NHAT trong phan ngoai 12 block cua backward (~1180 ms).
 * §0.133 ket luan "sat tran" dua tren 158 GFLOPS — nhung do lai thay tkv GEMM
 * dat 366 GFLOPS. Vay 515 ms nay con gi de lay?
 *
 * Kernel (tkv_simd_native.c L1690) co 5 pha:
 *   1. softmax-grad -> dl (T x V)   [malloc 206 MB]
 *   2. dxf = dl @ wte               [run_rect_shaped]
 *   3. tmp = dlT @ xf (trans-A)     [run_rect_ta_shaped] [malloc 154 MB]
 *   4. dwte += tmp                  [lmhead_add_rows]
 *   + malloc/free 360 MB moi lan goi
 *
 * Probe nay do TUNG pha de biet 515 ms di dau. Khong sua gi trong DLL.
 *
 * Build: powershell -ExecutionPolicy Bypass -File build_probe_lmhead_phases.ps1
 * Chay : ./probe_lmhead_phases.exe   (can tkv_simd_native.dll cung thu muc)
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>

/* 5 pha, export rieng de do */
typedef int  (*fused_t)(const float*, const int*, const float*, const float*,
                        float*, float*, int, int, int, int, float);
typedef void (*rect_t)(const float*, int, const float*, int, float*, int,
                       int, int, int, int, int, int, int, int, int, int);
typedef void (*rect_ta_t)(const float*, int, const float*, int, float*, int,
                          int, int, int, int, int, int, int, int, int, int);

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double best_of(int reps, double (*fn)(void*), void* ctx) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        fn(ctx);
        double t1 = now_ms();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

/* ---- context cho tung phep do ---- */
typedef struct {
    float *logits, *wte, *xf, *dxf, *dwte;
    int *ids;
    int T, E, V;
    float scale;
    fused_t fused;
} ctx_t;

static double run_fused(void* p) {
    ctx_t* c = (ctx_t*)p;
    return (double)c->fused(c->logits, c->ids, c->wte, c->xf, c->dxf, c->dwte,
                            1, c->T, c->E, c->V, c->scale);
}

/* pha 2: dxf = dl @ wte  — run_rect_shaped(dl,V, wte,E, dxf,E, T,E,V) */
typedef struct { float *dl, *wte, *dxf; int T, E, V; rect_t fn; } ctx2_t;
static double run_p2(void* p) {
    ctx2_t* c = (ctx2_t*)p;
    c->fn(c->dl, c->V, c->wte, c->E, c->dxf, c->E, c->T, c->E, c->V,
          0, c->T, 0, c->E, 0, 0, 0);
    return 0.0;
}

/* pha 3: tmp = dlT @ xf — run_rect_ta_shaped(dl,V, xf,E, tmp,E, V,E,T) */
typedef struct { float *dl, *xf, *tmp; int T, E, V; rect_ta_t fn; } ctx3_t;
static double run_p3(void* p) {
    ctx3_t* c = (ctx3_t*)p;
    c->fn(c->dl, c->V, c->xf, c->E, c->tmp, c->E, c->V, c->E, c->T,
          0, c->V, 0, c->E, 0, 0, 0);
    return 0.0;
}

/* malloc 206 MB (dl) */
static double run_malloc_dl(void* p) {
    int* n = (int*)p;
    float* q = (float*)malloc((size_t)(*n) * sizeof(float));
    if (q) { q[0] = 1.0f; free(q); }
    return 0.0;
}
/* malloc 154 MB (tmp) */
static double run_malloc_tmp(void* p) {
    int* n = (int*)p;
    float* q = (float*)malloc((size_t)(*n) * sizeof(float));
    if (q) { q[0] = 1.0f; free(q); }
    return 0.0;
}

int main(int argc, char** argv) {
    int T = 1024, E = 768, V = 50257;
    if (argc > 3) { T = atoi(argv[1]); E = atoi(argv[2]); V = atoi(argv[3]); }

    printf("=== lmhead_bwd_fused PHASE SPLIT  T=%d E=%d V=%d ===\n", T, E, V);
    printf("    dl  = T*V*4 = %.1f MB\n", (double)T * V * 4.0 / 1048576.0);
    printf("    tmp = V*E*4 = %.1f MB\n\n", (double)V * E * 4.0 / 1048576.0);

    HMODULE h = LoadLibraryA("tkv_simd_native.dll");
    if (!h) { printf("load DLL fail\n"); return 1; }

    fused_t fused = (fused_t)GetProcAddress(h, "lmhead_bwd_fused");
    rect_t p2 = (rect_t)GetProcAddress(h, "run_rect_shaped");
    rect_ta_t p3 = (rect_ta_t)GetProcAddress(h, "run_rect_ta_shaped");
    printf("sym lmhead_bwd_fused : %s\n", fused ? "OK" : "MISSING");
    printf("sym run_rect_shaped  : %s\n", p2 ? "OK" : "MISSING");
    printf("sym run_rect_ta_shaped: %s\n", p3 ? "OK" : "MISSING");
    if (!fused) return 1;

    size_t nlg = (size_t)T * V;
    size_t nwe = (size_t)V * E;
    size_t nxe = (size_t)T * E;

    float* logits = (float*)malloc(nlg * 4);
    float* wte    = (float*)malloc(nwe * 4);
    float* xf     = (float*)malloc(nxe * 4);
    float* dxf    = (float*)malloc(nxe * 4);
    float* dwte   = (float*)malloc(nwe * 4);
    int*   ids    = (int*)malloc((size_t)T * 4);
    if (!logits || !wte || !xf || !dxf || !dwte || !ids) { printf("malloc fail\n"); return 1; }

    for (size_t i = 0; i < nlg; i++) logits[i] = 0.001f * (float)(i % 97);
    for (size_t i = 0; i < nwe; i++) wte[i] = 0.001f;
    for (size_t i = 0; i < nxe; i++) xf[i] = 0.001f;
    for (int i = 0; i < T; i++) ids[i] = (i * 37 + 11) % V;
    float scale = 1.0f / (float)T;

    /* ---- A) toan bo kernel ---- */
    ctx_t c = { logits, wte, xf, dxf, dwte, ids, T, E, V, scale, fused };
    double t_fused = best_of(3, run_fused, &c);
    printf("A) FULL lmhead_bwd_fused      = %8.2f ms\n", t_fused);

    /* ---- B) chi malloc 2 buffer ---- */
    double t_m1 = best_of(5, run_malloc_dl, &T);
    double t_m2 = best_of(5, run_malloc_tmp, &V);
    printf("B) malloc dl  (%.0f MB)        = %8.2f ms\n", (double)T * V * 4.0 / 1048576.0, t_m1);
    printf("C) malloc tmp (%.0f MB)        = %8.2f ms\n", (double)V * E * 4.0 / 1048576.0, t_m2);
    printf("D) tong malloc 2 buffer       = %8.2f ms  (%.1f%% cua FULL)\n",
           t_m1 + t_m2, 100.0 * (t_m1 + t_m2) / t_fused);

    /* ---- E/F) pha 2 va pha 3 rieng ---- */
    float* dl = (float*)malloc(nlg * 4);
    float* tmp = (float*)malloc(nwe * 4);
    if (dl && tmp) {
        for (size_t i = 0; i < nlg; i++) dl[i] = 0.0001f;
        if (p2) {
            ctx2_t c2 = { dl, wte, dxf, T, E, V, p2 };
            double t2 = best_of(3, run_p2, &c2);
            double fl2 = 2.0 * (double)T * E * V;
            printf("E) pha2 dxf = dl @ wte        = %8.2f ms  -> %6.1f GFLOPS\n",
                   t2, fl2 / (t2 * 1e6));
        }
        if (p3) {
            ctx3_t c3 = { dl, xf, tmp, T, E, V, p3 };
            double t3 = best_of(3, run_p3, &c3);
            double fl3 = 2.0 * (double)T * E * V;
            printf("F) pha3 tmp = dlT @ xf (ta)   = %8.2f ms  -> %6.1f GFLOPS\n",
                   t3, fl3 / (t3 * 1e6));
        }
    }

    printf("\n--- ket luan ---\n");
    printf("FULL %.2f ms; neu malloc %.2f + pha2 %.2f + pha3 %.2f thi con lai la softmax+add\n",
           t_fused, t_m1 + t_m2, 0.0, 0.0);
    return 0;
}
