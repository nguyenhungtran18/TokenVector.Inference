/* probe_lmhead_pha3.c - PROBE. Pha3 `tmp[V,E] = dlT @ xf` chay 301 GFLOP/s
 * trong khi pha2 cung 39.5 GFLOP chay 561. Tim bien the nhanh hon.
 *
 * Pha3 hien tai: cblas_sgemm(RowMajor, TransA, NoTrans, m=V, n=E, k=T,
 *                            A=dl (lda=V), B=xf (ldb=E), C=tmp (ldc=E))
 *   -> A doc theo COT, stride T*4 = 4096 B. Day la nghi pham.
 *
 * Cac bien the do:
 *   A) cblas_sgemm TransA (hien tai)
 *   B) cblas_sgemm NoTrans tren dlT da materialize (copy 196 MB)
 *   C) ATen mm(dl.t(), xf) — oneDNN lo transpose
 *   D) ATen mm tren dlT contiguous
 *   E) chi phi copy dlT (de biet B co lai khong)
 *
 * Build: build_probe_lmhead_pha3.ps1
 * Chay:  ./probe_lmhead_pha3.exe 1024 768 50257
 */
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

static double now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

/* ---- cblas qua scipy-openblas (giong production) ---- */
typedef void (*sgemm_t)(int Order, int TransA, int TransB, int M, int N, int K,
                        float alpha, const float* A, int lda,
                        const float* B, int ldb, float beta, float* C, int ldc);
typedef void (*setnt_t)(int);

static HMODULE g_blas = NULL;
static sgemm_t p_sgemm = NULL;
static setnt_t p_setnt = NULL;

static int load_blas(void) {
    const char* names[] = {
        "libscipy_openblas64_.dll", "libopenblas64_.dll",
        "libscipy_openblas.dll", "openblas.dll", NULL
    };
    for (int i = 0; names[i]; i++) {
        HMODULE h = LoadLibraryA(names[i]);
        if (h) {
            p_sgemm = (sgemm_t)GetProcAddress(h, "scipy_cblas_sgemm");
            if (!p_sgemm) p_sgemm = (sgemm_t)GetProcAddress(h, "cblas_sgemm");
            p_setnt = (setnt_t)GetProcAddress(h, "scipy_openblas_set_num_threads");
            if (!p_setnt) p_setnt = (setnt_t)GetProcAddress(h, "openblas_set_num_threads");
            if (p_sgemm) { g_blas = h; return 1; }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int T = (argc > 1) ? atoi(argv[1]) : 1024;
    int E = (argc > 2) ? atoi(argv[2]) : 768;
    int V = (argc > 3) ? atoi(argv[3]) : 50257;
    int reps = (argc > 4) ? atoi(argv[4]) : 5;

    size_t nlg = (size_t)T * V;      /* dl   */
    size_t nwe = (size_t)V * E;      /* tmp  */
    size_t nxe = (size_t)T * E;      /* xf   */
    double fl = 2.0 * (double)T * E * V;

    printf("=== pha3 tmp[V,E] = dlT[T,V]^T @ xf[T,E] ===\n");
    printf("    T=%d E=%d V=%d   dl=%.1f MB  tmp=%.1f MB  %.1f GFLOP\n",
           T, E, V, nlg * 4 / 1048576.0, nwe * 4 / 1048576.0, fl / 1e9);

    int blas = load_blas();
    printf("    BLAS=%d\n\n", blas);

    float* dl   = (float*)malloc(nlg * 4);
    float* dlT  = (float*)malloc(nlg * 4);
    float* xf   = (float*)malloc(nxe * 4);
    float* tmp  = (float*)malloc(nwe * 4);
    float* tmp2 = (float*)malloc(nwe * 4);
    if (!dl || !dlT || !xf || !tmp || !tmp2) { printf("malloc fail\n"); return 1; }
    for (size_t i = 0; i < nlg; i++) dl[i] = 0.001f * (float)(i % 97);
    for (size_t i = 0; i < nxe; i++) xf[i] = 0.001f * (float)(i % 89);

    auto opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto t_dl = torch::from_blob(dl, {T, V}, opt);
    auto t_xf = torch::from_blob(xf, {T, E}, opt);
    auto t_tmp = torch::from_blob(tmp2, {V, E}, opt);

    /* ---------- E) chi phi copy dlT ---------- */
    double bestE = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        for (int v = 0; v < V; v++)
            for (int t = 0; t < T; t++)
                dlT[(size_t)v * T + t] = dl[(size_t)t * V + v];
        double t = now_ms() - t0;
        if (t < bestE) bestE = t;
    }
    printf("E) copy dl -> dlT (196 MB, scalar loop)  = %8.2f ms\n", bestE);

    /* ---------- A) cblas TransA (hien tai) ---------- */
    double bestA = 1e30;
    if (blas) {
        if (p_setnt) p_setnt(28);
        p_sgemm(101, 112, 111, V, E, T, 1.0f, dl, V, xf, E, 0.0f, tmp, E);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            p_sgemm(101, 112, 111, V, E, T, 1.0f, dl, V, xf, E, 0.0f, tmp, E);
            double t = now_ms() - t0;
            if (t < bestA) bestA = t;
        }
        printf("A) cblas sgemm TransA (hien tai)         = %8.2f ms  -> %6.1f GFLOPS\n",
               bestA, fl / (bestA * 1e6));

        /* ---------- B) cblas NoTrans tren dlT ---------- */
        double bestB = 1e30;
        p_sgemm(101, 111, 111, V, E, T, 1.0f, dlT, T, xf, E, 0.0f, tmp, E);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            p_sgemm(101, 111, 111, V, E, T, 1.0f, dlT, T, xf, E, 0.0f, tmp, E);
            double t = now_ms() - t0;
            if (t < bestB) bestB = t;
        }
        printf("B) cblas sgemm NoTrans tren dlT          = %8.2f ms  -> %6.1f GFLOPS\n",
               bestB, fl / (bestB * 1e6));
        printf("   B + copy = %8.2f ms   (lai so voi A: %+.2f ms)\n",
               bestB + bestE, bestA - (bestB + bestE));
    } else {
        printf("A) B) BO QUA - khong nap duoc BLAS\n");
    }

    /* ---------- C) ATen mm(dl.t(), xf) ---------- */
    double bestC = 1e30;
    int rcC = 0;
    try {
        auto y = at::mm(t_dl.t(), t_xf);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            auto y2 = at::mm(t_dl.t(), t_xf);
            double t = now_ms() - t0;
            if (t < bestC) bestC = t;
        }
        printf("C) ATen mm(dl.t(), xf)                   = %8.2f ms  -> %6.1f GFLOPS\n",
               bestC, fl / (bestC * 1e6));
    } catch (const std::exception& e) {
        rcC = -1;
        printf("C) ATen mm(dl.t(), xf)  LOI: %s\n", e.what());
    }

    /* ---------- D) ATen mm tren dlT contiguous ---------- */
    double bestD = 1e30;
    try {
        auto t_dlT = torch::from_blob(dlT, {V, T}, opt);
        auto y = at::mm(t_dlT, t_xf);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            auto y2 = at::mm(t_dlT, t_xf);
            double t = now_ms() - t0;
            if (t < bestD) bestD = t;
        }
        printf("D) ATen mm(dlT contiguous, xf)           = %8.2f ms  -> %6.1f GFLOPS\n",
               bestD, fl / (bestD * 1e6));
        printf("   D + copy = %8.2f ms   (lai so voi A: %+.2f ms)\n",
               bestD + bestE, bestA - (bestD + bestE));
    } catch (const std::exception& e) {
        printf("D) ATen mm(dlT) LOI: %s\n", e.what());
    }

    /* ---------- verify: A va C phai khop ---------- */
    if (blas) {
        double md = 0.0;
        for (size_t i = 0; i < nwe; i++) {
            double d = fabs((double)tmp[i] - (double)tmp2[i]);
            if (d > md) md = d;
        }
        printf("\nmax_abs(A - C) = %.6e   (|tmp| tong = ", md);
        double s = 0.0;
        for (size_t i = 0; i < nwe; i++) s += fabs((double)tmp[i]);
        printf("%.3f)\n", s);
    }

    /* ---------- F) ATen addmm beta=1: gop pha3+pha4, bo buffer tmp ---------- */
    /* dwte += dlT @ xf  ->  ATen addmm(dwte, dl.t(), xf, beta=1, alpha=1) */
    {
        auto t_dw = torch::from_blob(tmp, {V, E}, opt);   /* dung lam dwte tam */
        double bestF = 1e30;
        try {
            auto y = at::addmm(t_dw, t_dl.t(), t_xf, 1.0, 1.0);
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                auto y2 = at::addmm(t_dw, t_dl.t(), t_xf, 1.0, 1.0);
                double t = now_ms() - t0;
                if (t < bestF) bestF = t;
            }
            printf("F) ATen addmm(beta=1) gop pha3+pha4       = %8.2f ms  -> %6.1f GFLOPS\n",
                   bestF, fl / (bestF * 1e6));
            printf("   so voi A + pha4(36.5) = %8.2f ms   lai: %+.2f ms\n",
                   258.46 + 36.5, (258.46 + 36.5) - bestF);
        } catch (const std::exception& e) {
            printf("F) addmm LOI: %s\n", e.what());
        }
    }

    /* ---------- G) ATen voi so thread khac nhau ---------- */
    {
        int ths[] = {14, 20, 28};
        for (int k = 0; k < 3; k++) {
            at::set_num_threads(ths[k]);
            double b = 1e30;
            auto y = at::mm(t_dl.t(), t_xf);
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                auto y2 = at::mm(t_dl.t(), t_xf);
                double t = now_ms() - t0;
                if (t < b) b = t;
            }
            printf("G) ATen mm  %2d thread                    = %8.2f ms  -> %6.1f GFLOPS\n",
                   ths[k], b, fl / (b * 1e6));
        }
        at::set_num_threads(28);
    }

    /* ---------- H) dang transpose: tmpT[E,V] = xfT @ dl ---------- */
    {
        double b = 1e30;
        try {
            auto y = at::mm(t_xf.t(), t_dl);
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                auto y2 = at::mm(t_xf.t(), t_dl);
                double t = now_ms() - t0;
                if (t < b) b = t;
            }
            printf("H) ATen mm(xf.t(), dl) -> tmpT[E,V]       = %8.2f ms  -> %6.1f GFLOPS\n",
                   b, fl / (b * 1e6));
        } catch (const std::exception& e) {
            printf("H) LOI: %s\n", e.what());
        }
    }

    /* ---------- I) openblas theo so thread (production dung tkv_nthreads) ---------- */
    if (blas && p_setnt) {
        int ths[] = {7, 14, 20, 28};
        printf("\nI) openblas sgemm TransA theo so thread:\n");
        for (int k = 0; k < 4; k++) {
            p_setnt(ths[k]);
            double b = 1e30;
            p_sgemm(101, 112, 111, V, E, T, 1.0f, dl, V, xf, E, 0.0f, tmp, E);
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                p_sgemm(101, 112, 111, V, E, T, 1.0f, dl, V, xf, E, 0.0f, tmp, E);
                double t = now_ms() - t0;
                if (t < b) b = t;
            }
            printf("   %2d thread = %8.2f ms  -> %6.1f GFLOPS\n", ths[k], b, fl / (b * 1e6));
        }
        p_setnt(28);
    }

    /* ---------- J) pha2 dxf = dl @ wte, cung thread de so ---------- */
    if (blas && p_setnt) {
        float* dxf = (float*)malloc(nxe * 4);
        float* wte = (float*)malloc(nwe * 4);
        if (dxf && wte) {
            for (size_t i = 0; i < nwe; i++) wte[i] = 0.001f;
            printf("\nJ) pha2 dxf[T,E] = dl @ wte  (m=T n=E k=V):\n");
            int ths[] = {7, 14, 20, 28};
            for (int k = 0; k < 4; k++) {
                p_setnt(ths[k]);
                double b = 1e30;
                p_sgemm(101, 111, 111, T, E, V, 1.0f, dl, V, wte, E, 0.0f, dxf, E);
                for (int r = 0; r < reps; r++) {
                    double t0 = now_ms();
                    p_sgemm(101, 111, 111, T, E, V, 1.0f, dl, V, wte, E, 0.0f, dxf, E);
                    double t = now_ms() - t0;
                    if (t < b) b = t;
                }
                printf("   %2d thread = %8.2f ms  -> %6.1f GFLOPS\n", ths[k], b, fl / (b * 1e6));
            }
            p_setnt(28);
        }
        if (dxf) free(dxf);
        if (wte) free(wte);
    }

    /* ---------- K) ATen tren shape pha2, quet thread ---------- */
    {
        float* wte = (float*)malloc(nwe * 4);
        if (wte) {
            for (size_t i = 0; i < nwe; i++) wte[i] = 0.001f;
            auto t_wte = torch::from_blob(wte, {V, E}, opt);
            printf("\nK) ATen mm(dl, wte) = pha2 shape, quet thread:\n");
            int ths[] = {7, 14, 20, 28};
            for (int k = 0; k < 4; k++) {
                at::set_num_threads(ths[k]);
                double b = 1e30;
                auto y = at::mm(t_dl, t_wte);
                for (int r = 0; r < reps; r++) {
                    double t0 = now_ms();
                    auto y2 = at::mm(t_dl, t_wte);
                    double t = now_ms() - t0;
                    if (t < b) b = t;
                }
                printf("   %2d thread = %8.2f ms  -> %6.1f GFLOPS\n", ths[k], b, fl / (b * 1e6));
            }
            at::set_num_threads(28);
            free(wte);
        }
    }

    /* ---------- L) pha3 chia V thanh nhieu manh, ATen 14 thread ---------- */
    {
        printf("\nL) pha3 cat V thanh k manh (ATen 14 thr):\n");
        at::set_num_threads(14);
        int splits[] = {1, 2, 4};
        for (int s = 0; s < 3; s++) {
            int ks = splits[s];
            int vb = V / ks;
            double b = 1e30;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                for (int j = 0; j < ks; j++) {
                    int v0 = j * vb;
                    int v1 = (j == ks - 1) ? V : (j + 1) * vb;
                    auto a = t_dl.t().narrow(0, v0, v1 - v0);
                    auto y = at::mm(a, t_xf);
                }
                double t = now_ms() - t0;
                if (t < b) b = t;
            }
            printf("   %d manh (moi manh ~%d hang) = %8.2f ms  -> %6.1f GFLOPS\n",
                   ks, vb, b, fl / (b * 1e6));
        }
        at::set_num_threads(28);
    }

    /* ---------- M) mm_out / addmm_out: ghi vao buffer co san ---------- */
    {
        printf("\nM) ghi vao buffer co san (ATen 14 thr):\n");
        at::set_num_threads(14);
        double b = 1e30;
        auto out_t = torch::from_blob(tmp, {V, E}, opt);
        auto y = at::mm_out(out_t, t_dl.t(), t_xf);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            auto y2 = at::mm_out(out_t, t_dl.t(), t_xf);
            double t = now_ms() - t0;
            if (t < b) b = t;
        }
        printf("   mm_out(tmp, dl.t(), xf)          = %8.2f ms  -> %6.1f GFLOPS\n",
               b, fl / (b * 1e6));

        double b2 = 1e30;
        auto dw_t = torch::from_blob(tmp2, {V, E}, opt);
        auto z = at::addmm_out(dw_t, dw_t, t_dl.t(), t_xf, 1.0, 1.0);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            auto z2 = at::addmm_out(dw_t, dw_t, t_dl.t(), t_xf, 1.0, 1.0);
            double t = now_ms() - t0;
            if (t < b2) b2 = t;
        }
        printf("   addmm_out(dwte, dwte, dl.t(), xf) = %8.2f ms  -> %6.1f GFLOPS\n",
               b2, fl / (b2 * 1e6));
        printf("   (production: pha3 258 + pha4 36.5 = 294.5 ms)\n");
        at::set_num_threads(28);
    }
    return 0;
}
