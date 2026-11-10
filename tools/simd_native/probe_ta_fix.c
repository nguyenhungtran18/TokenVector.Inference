/* probe_ta_fix.c - trans-A GEMM 255 ms: co cach nao nhanh hon khong?
 *
 * Do duoc (probe_lmhead_phases3, production-mirror, BLAS=openblas, 28 thr):
 *   pha2  dxf = dl  @ wte   [1024,50257]x[50257,768]  NoTrans  = 142 ms  557 GFLOPS
 *   pha3  tmp = dlT @ xf    [50257,1024]x[1024,768]   transA   = 255 ms  309 GFLOPS
 * Cung so FLOP (39.5 GFLOP) nhung trans-A cham 1.80x.
 *
 * Gia thuyet: transA=112 bat openblas doc A = dl [T][V] theo chieu cot, stride
 * V*4 = 201 KB moi ptu -> TLB/cache tham hoai. Neu transpose dl thanh dlT
 * [V][T] lien mach roi chay NoTrans thi co nhanh hon (transpose + gemm) khong?
 *
 * Do 4 phuong an tren cung shape:
 *   P1  transA truc tiep (hien tai)                     = baseline
 *   P2  transpose dl->dlT  (do rieng)
 *   P3  NoTrans tren dlT    (do rieng)
 *   P4  P2 + P3 (tong)  <-- so voi P1
 *
 * Build: powershell -ExecutionPolicy Bypass -File build_probe_ta_fix.ps1
 * Chay : ./probe_ta_fix.exe   (can libscipy_openblas64_.dll cung thu muc)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c"

static double now_ms4(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* transpose A [k][m] -> At [m][k], blocked de khong tham hoai cache */
static void transpose_blocked(const float* a, float* at, int k, int m) {
    const int BS = 32;
    for (int i0 = 0; i0 < k; i0 += BS) {
        int i1 = i0 + BS; if (i1 > k) i1 = k;
        for (int j0 = 0; j0 < m; j0 += BS) {
            int j1 = j0 + BS; if (j1 > m) j1 = m;
            for (int i = i0; i < i1; i++) {
                const float* src = a + (long long)i * m;
                for (int j = j0; j < j1; j++) at[(long long)j * k + i] = src[j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int T = 1024, E = 768, V = 50257;
    if (argc > 3) { T = atoi(argv[1]); E = atoi(argv[2]); V = atoi(argv[3]); }

    printf("=== trans-A GEMM: tim phuong an nhanh hon ===\n");
    printf("    T=%d E=%d V=%d   BLAS=%d  thr=%d\n\n", T, E, V,
           tkv_blas_load(), tkv_nthreads(V));

    size_t nlg = (size_t)T * V;   /* dl  [T][V] */
    size_t nwe = (size_t)V * E;   /* tmp [V][E] */
    size_t nxe = (size_t)T * E;   /* xf  [T][E] */

    float* dl  = (float*)malloc(nlg * 4);
    float* dlT = (float*)malloc(nlg * 4);
    float* xf  = (float*)malloc(nxe * 4);
    float* tmp = (float*)malloc(nwe * 4);
    if (!dl || !dlT || !xf || !tmp) { printf("malloc fail\n"); return 1; }
    for (size_t i = 0; i < nlg; i++) dl[i] = 0.0001f * (float)(i % 89);
    for (size_t i = 0; i < nxe; i++) xf[i] = 0.001f;
    for (size_t i = 0; i < nwe; i++) tmp[i] = 0.0f;

    double fl = 2.0 * (double)T * E * V;

    /* ---------- P1: transA truc tiep (hien tai) ---------- */
    double p1 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms4();
        run_rect_ta_shaped(dl, V, xf, E, tmp, E, V, E, T);
        double t1 = now_ms4();
        if (t1 - t0 < p1) p1 = t1 - t0;
    }
    printf("P1) transA truc tiep (hien tai)   = %8.2f ms  -> %6.1f GFLOPS\n",
           p1, fl / (p1 * 1e6));

    /* ---------- P2: transpose dl -> dlT ---------- */
    double p2 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms4();
        transpose_blocked(dl, dlT, T, V);
        double t1 = now_ms4();
        if (t1 - t0 < p2) p2 = t1 - t0;
    }
    printf("P2) transpose dl -> dlT           = %8.2f ms  (%.0f MB doc + %.0f MB ghi)\n",
           p2, (double)nlg * 4 / 1048576.0, (double)nlg * 4 / 1048576.0);

    /* ---------- P3: NoTrans tren dlT ---------- */
    double p3 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms4();
        run_rect_shaped(dlT, T, xf, E, tmp, E, V, E, T);
        double t1 = now_ms4();
        if (t1 - t0 < p3) p3 = t1 - t0;
    }
    printf("P3) NoTrans tren dlT              = %8.2f ms  -> %6.1f GFLOPS\n",
           p3, fl / (p3 * 1e6));

    /* ---------- P4: P2 + P3 ---------- */
    double p4 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms4();
        transpose_blocked(dl, dlT, T, V);
        run_rect_shaped(dlT, T, xf, E, tmp, E, V, E, T);
        double t1 = now_ms4();
        if (t1 - t0 < p4) p4 = t1 - t0;
    }
    printf("P4) P2 + P3 (tong)                = %8.2f ms  -> %6.1f GFLOPS\n",
           p4, fl / (p4 * 1e6));

    /* ---------- P5: transpose + transA tren dlT (doi chieu) ---------- */
    double p5 = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_ms4();
        run_rect_ta_shaped(dlT, T, xf, E, tmp, E, V, E, T);
        double t1 = now_ms4();
        if (t1 - t0 < p5) p5 = t1 - t0;
    }
    printf("P5) transA tren dlT (doi chieu)   = %8.2f ms\n", p5);

    printf("\n--- KET LUAN ---\n");
    printf("P1 (hien tai)        = %8.2f ms\n", p1);
    printf("P4 (transpose+notr)  = %8.2f ms\n", p4);
    if (p4 < p1)
        printf("=> P4 NHANH HON %.2fx, tiet kiem %.0f ms/buoc\n", p1 / p4, p1 - p4);
    else
        printf("=> P4 CHAM HON %.2fx, KHONG dung duoc (transpose %.0f ms qua dat)\n",
               p4 / p1, p2);
    return 0;
}
