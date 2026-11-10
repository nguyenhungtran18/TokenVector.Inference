/* probe_lmhead_phases3.c - tach 5 pha cua lmhead_bwd_fused, KHOP PRODUCTION
 *
 * probe_lmhead_phases2 do softmax 1 thread (651 ms) va GEMM native -> khong
 * khop 515 ms cua .tkv. Nguyen nhan:
 *   - production goi tkv_run_aux(nt, 3, ...) cho softmax (28 thread), khong
 *     goi lmhead_softmax_rows truc tiep
 *   - production goi tkv_run_aux(nta, 5, ...) cho dwte += tmp
 *   - production nap scipy-openblas64_ cho GEMM (tkv_blas_load)
 * Probe nay mirror y nguyen 5 pha cua lmhead_bwd_fused (L1690).
 *
 * Build: powershell -ExecutionPolicy Bypass -File build_probe_lmhead_phases3.ps1
 * Chay : ./probe_lmhead_phases3.exe   (can libscipy_openblas64_.dll cung thu muc)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c"

static double now_ms3(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int T = 1024, E = 768, V = 50257;
    if (argc > 3) { T = atoi(argv[1]); E = atoi(argv[2]); V = atoi(argv[3]); }

    size_t nlg = (size_t)T * V;
    size_t nwe = (size_t)V * E;
    size_t nxe = (size_t)T * E;

    int blas = tkv_blas_load();
    printf("=== lmhead_bwd_fused PHASE SPLIT (production-mirror) ===\n");
    printf("    T=%d E=%d V=%d\n", T, E, V);
    printf("    dl=%.1f MB  tmp=%.1f MB\n", (double)nlg * 4 / 1048576.0, (double)nwe * 4 / 1048576.0);
    printf("    BLAS=%d  threads(T)=%d  threads(V)=%d\n\n",
           blas, tkv_nthreads(T), tkv_nthreads(V));

    float* logits = (float*)malloc(nlg * 4);
    float* wte    = (float*)malloc(nwe * 4);
    float* xf     = (float*)malloc(nxe * 4);
    float* dxf    = (float*)malloc(nxe * 4);
    float* dwte   = (float*)malloc(nwe * 4);
    int*   ids    = (int*)malloc((size_t)T * 4);
    float* dl     = (float*)malloc(nlg * 4);
    float* tmp    = (float*)malloc(nwe * 4);
    if (!logits || !wte || !xf || !dxf || !dwte || !ids || !dl || !tmp) {
        printf("malloc fail\n"); return 1;
    }
    for (size_t i = 0; i < nlg; i++) logits[i] = 0.001f * (float)(i % 97);
    for (size_t i = 0; i < nwe; i++) wte[i] = 0.001f;
    for (size_t i = 0; i < nxe; i++) xf[i] = 0.001f;
    for (int i = 0; i < T; i++) ids[i] = (i * 37 + 11) % V;
    float scale = 1.0f / (float)T;

    int nt = tkv_nthreads(T);
    int nta = tkv_nthreads(V);

    /* ---------- A) FULL ---------- */
    double best_full = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms3();
        int st = lmhead_bwd_fused(logits, ids, wte, xf, dxf, dwte, 1, T, E, V, scale);
        double t1 = now_ms3();
        if (st != 0) { printf("fused st=%d\n", st); return 1; }
        if (t1 - t0 < best_full) best_full = t1 - t0;
    }
    printf("A) FULL lmhead_bwd_fused           = %8.2f ms\n", best_full);

    /* ---------- B) pha1 softmax SONG SONG (dung nhu production) ---------- */
    double bs = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms3();
        if (nt <= 1) lmhead_softmax_rows(logits, ids, dl, V, 0, T, scale);
        else tkv_run_aux(nt, 3, logits, ids, dl, V, 0, scale, T);
        double t1 = now_ms3();
        if (t1 - t0 < bs) bs = t1 - t0;
    }
    printf("B) pha1 softmax-grad (%2d thr)      = %8.2f ms\n", nt, bs);

    /* ---------- C) pha2 dxf = dl @ wte ---------- */
    double b2 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms3();
        run_rect_shaped(dl, V, wte, E, dxf, E, T, E, V);
        double t1 = now_ms3();
        if (t1 - t0 < b2) b2 = t1 - t0;
    }
    double fl = 2.0 * (double)T * E * V;
    printf("C) pha2 dxf = dl @ wte             = %8.2f ms  -> %6.1f GFLOPS\n", b2, fl / (b2 * 1e6));

    /* ---------- D) pha3 tmp = dlT @ xf (trans-A) ---------- */
    double b3 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms3();
        run_rect_ta_shaped(dl, V, xf, E, tmp, E, V, E, T);
        double t1 = now_ms3();
        if (t1 - t0 < b3) b3 = t1 - t0;
    }
    printf("D) pha3 tmp = dlT @ xf (trans-A)   = %8.2f ms  -> %6.1f GFLOPS\n", b3, fl / (b3 * 1e6));

    /* ---------- E) pha4 dwte += tmp (song song nhu production) ---------- */
    double b4 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms3();
        if (nta <= 1) lmhead_add_rows(dwte, tmp, E, 0, V);
        else tkv_run_aux(nta, 5, tmp, NULL, dwte, E, 0, 0.0f, V);
        double t1 = now_ms3();
        if (t1 - t0 < b4) b4 = t1 - t0;
    }
    printf("E) pha4 dwte += tmp (%2d thr)       = %8.2f ms\n", nta, b4);

    /* ---------- F) malloc 2 buffer ---------- */
    double bm = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = now_ms3();
        float* q1 = (float*)malloc(nlg * 4);
        float* q2 = (float*)malloc(nwe * 4);
        if (q1) { q1[0] = 1.0f; free(q1); }
        if (q2) { q2[0] = 1.0f; free(q2); }
        double t1 = now_ms3();
        if (t1 - t0 < bm) bm = t1 - t0;
    }
    printf("F) malloc dl+tmp                   = %8.2f ms\n", bm);

    double sum = bm + bs + b2 + b3 + b4;
    printf("\n--- DOI CHIEU (production-mirror) ---\n");
    printf("  malloc          %8.2f ms  (%5.1f%%)\n", bm, 100.0 * bm / best_full);
    printf("  pha1 softmax    %8.2f ms  (%5.1f%%)\n", bs, 100.0 * bs / best_full);
    printf("  pha2 dxf gemm   %8.2f ms  (%5.1f%%)\n", b2, 100.0 * b2 / best_full);
    printf("  pha3 ta gemm    %8.2f ms  (%5.1f%%)\n", b3, 100.0 * b3 / best_full);
    printf("  pha4 dwte add   %8.2f ms  (%5.1f%%)\n", b4, 100.0 * b4 / best_full);
    printf("  --------------------------------\n");
    printf("  SUM             %8.2f ms\n", sum);
    printf("  FULL            %8.2f ms\n", best_full);
    printf("  CHENH           %8.2f ms  (%.1f%%)\n", best_full - sum,
           100.0 * (best_full - sum) / best_full);
    printf("\n  GEMM2+GEMM3 = %.2f ms = %.1f%% cua FULL\n", b2 + b3, 100.0 * (b2 + b3) / best_full);
    printf("  softmax     = %.2f ms = %.1f%% cua FULL\n", bs, 100.0 * bs / best_full);
    return 0;
}
