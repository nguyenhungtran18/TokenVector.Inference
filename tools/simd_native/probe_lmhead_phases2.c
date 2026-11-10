/* probe_lmhead_phases2.c - tach 5 pha cua lmhead_bwd_fused
 *
 * run_rect_shaped / run_rect_ta_shaped la `static` trong tkv_simd_native.c nen
 * khong goi duoc qua DLL. Cach sach: #include nguyen TU do vao probe nay roi
 * goi truc tiep. KHONG sua dong nao trong source production.
 *
 * Build: powershell -ExecutionPolicy Bypass -File build_probe_lmhead_phases2.ps1
 * Chay : ./probe_lmhead_phases2.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* nguyen TU production - de goi duoc ham static */
#include "D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c"

static double now_ms2(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int T = 1024, E = 768, V = 50257;
    if (argc > 3) { T = atoi(argv[1]); E = atoi(argv[2]); V = atoi(argv[3]); }

    size_t nlg = (size_t)T * V;
    size_t nwe = (size_t)V * E;
    size_t nxe = (size_t)T * E;

    printf("=== lmhead_bwd_fused PHASE SPLIT  T=%d E=%d V=%d ===\n", T, E, V);
    printf("    dl  = T*V*4 = %.1f MB\n", (double)nlg * 4.0 / 1048576.0);
    printf("    tmp = V*E*4 = %.1f MB\n", (double)nwe * 4.0 / 1048576.0);
    printf("    threads = %d\n\n", tkv_nthreads(T));

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

    /* ---------- A) FULL kernel ---------- */
    double best_full = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms2();
        int st = lmhead_bwd_fused(logits, ids, wte, xf, dxf, dwte, 1, T, E, V, scale);
        double t1 = now_ms2();
        if (st != 0) { printf("fused st=%d\n", st); return 1; }
        if (t1 - t0 < best_full) best_full = t1 - t0;
    }
    printf("A) FULL lmhead_bwd_fused        = %8.2f ms\n", best_full);

    /* ---------- B/C) malloc 2 buffer ---------- */
    double bm1 = 1e30, bm2 = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = now_ms2();
        float* q = (float*)malloc(nlg * 4);
        if (q) { q[0] = 1.0f; free(q); }
        double t1 = now_ms2();
        if (t1 - t0 < bm1) bm1 = t1 - t0;
    }
    for (int r = 0; r < 5; r++) {
        double t0 = now_ms2();
        float* q = (float*)malloc(nwe * 4);
        if (q) { q[0] = 1.0f; free(q); }
        double t1 = now_ms2();
        if (t1 - t0 < bm2) bm2 = t1 - t0;
    }
    printf("B) malloc dl  (%.0f MB)          = %8.2f ms\n", (double)nlg * 4.0 / 1048576.0, bm1);
    printf("C) malloc tmp (%.0f MB)          = %8.2f ms\n", (double)nwe * 4.0 / 1048576.0, bm2);
    printf("D) tong malloc 2 buffer         = %8.2f ms  (%.1f%% FULL)\n",
           bm1 + bm2, 100.0 * (bm1 + bm2) / best_full);

    /* ---------- E) pha 1: softmax-grad ---------- */
    float* dl = (float*)malloc(nlg * 4);
    if (!dl) { printf("malloc dl fail\n"); return 1; }
    double bs = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms2();
        lmhead_softmax_rows(logits, ids, dl, V, 0, T, scale);
        double t1 = now_ms2();
        if (t1 - t0 < bs) bs = t1 - t0;
    }
    printf("E) pha1 softmax-grad (1 thread) = %8.2f ms\n", bs);

    /* ---------- F) pha 2: dxf = dl @ wte ---------- */
    double b2 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms2();
        run_rect_shaped(dl, V, wte, E, dxf, E, T, E, V);
        double t1 = now_ms2();
        if (t1 - t0 < b2) b2 = t1 - t0;
    }
    double fl = 2.0 * (double)T * E * V;
    printf("F) pha2 dxf = dl @ wte          = %8.2f ms  -> %6.1f GFLOPS\n", b2, fl / (b2 * 1e6));

    /* ---------- G) pha 3: tmp = dlT @ xf (trans-A) ---------- */
    float* tmp = (float*)malloc(nwe * 4);
    if (!tmp) { printf("malloc tmp fail\n"); return 1; }
    double b3 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms2();
        run_rect_ta_shaped(dl, V, xf, E, tmp, E, V, E, T);
        double t1 = now_ms2();
        if (t1 - t0 < b3) b3 = t1 - t0;
    }
    printf("G) pha3 tmp = dlT @ xf (ta)     = %8.2f ms  -> %6.1f GFLOPS\n", b3, fl / (b3 * 1e6));

    /* ---------- H) pha 4: dwte += tmp ---------- */
    double b4 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms2();
        lmhead_add_rows(dwte, tmp, E, 0, V);
        double t1 = now_ms2();
        if (t1 - t0 < b4) b4 = t1 - t0;
    }
    printf("H) pha4 dwte += tmp (1 thread)  = %8.2f ms\n", b4);

    /* ---------- I) tong khop ---------- */
    double sum = bm1 + bm2 + bs + b2 + b3 + b4;
    printf("\n--- DOI CHIEU ---\n");
    printf("malloc %.2f + softmax %.2f + gemm2 %.2f + gemm3 %.2f + add %.2f\n",
           bm1 + bm2, bs, b2, b3, b4);
    printf("SUM pha  = %8.2f ms\n", sum);
    printf("FULL     = %8.2f ms\n", best_full);
    printf("CHENH    = %8.2f ms (%.1f%%)\n", best_full - sum,
           100.0 * (best_full - sum) / best_full);
    printf("\nGEMM2+GEMM3 = %.2f ms = %.1f%% cua FULL\n", b2 + b3, 100.0 * (b2 + b3) / best_full);
    return 0;
}
