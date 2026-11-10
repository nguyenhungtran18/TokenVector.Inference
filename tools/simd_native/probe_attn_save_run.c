/* probe_attn_save_run.c - PROBE. Do "luu ctx+lse o fwd" co tiet kiem that khong.
 *
 * Cau hoi dung KHONG phai "A vs B+C" (ca hai deu fwd+bwd mot lan). Cho tra
 * tien hai lan nam o CAP MODEL:
 *   hien tai : gpt2_attention (fwd, ATen sdpa, KHONG graph)  21.9 ms
 *            + TkvTorchAttentionBwd (dung lai graph fwd ben trong) 88.5 ms
 *   de xuat  : fwd co graph (B)  +  bwd dung lai graph (C)
 *
 * Do ca 4 bo weight (36 MB > L3 15 MB) cho khop dieu kien production.
 *
 * Build: build_probe_attn_save_run.ps1
 * Chay:  ./probe_attn_save_run.exe 1024 10
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef int (*fwdone_t)(const float*, const float*, const float*, const float*,
                        const float*, const float*, float*, float*,
                        long long, long long, long long, long long,
                        long long, long long, long long, long long,
                        double*, double*);
typedef int (*fwdsave_t)(const float*, const float*, const float*,
                         const float*, const float*, float*,
                         long long, long long, long long, long long, double*);
typedef int (*bwdsaved_t)(const float*, float*, float*, long long, long long,
                          long long, long long, long long, long long,
                          long long, double*);
typedef int (*valid_t)(void);
typedef int (*nodes_t)(void);
typedef int (*atnb_t)(const float*, const float*, const float*, const float*,
                      const float*, const float*, float*, float*,
                      long long, long long, long long, long long,
                      long long, long long, long long, long long);
typedef int (*sdpa_t)(const float*, float*, long long, long long, long long);
typedef void (*clear_t)(void);

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define NSET 4

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    int h = 768, hd = 64, nh = h / hd;
    int reps = (argc > 2) ? atoi(argv[2]) : 10;
    int nset = (argc > 3) ? atoi(argv[3]) : 4;
    if (nset < 1) nset = 1; if (nset > 12) nset = 12;

    long long npar = 124439808LL;                 /* gflat production */
    long long base = 50257LL * h + 1024LL * h + h + h;
    long long stride = h + h + 3LL * h * h + 3 * h + 1LL * h * h + h + h + h
                     + 4LL * h * h + 4 * h + 4LL * h * h + h;

    /* 4 bo weight THAT, moi bo o offset rieng trong gflat */
    float* wq[NSET]; float* wp[NSET]; float* bq[NSET]; float* bp[NSET];
    long long oq[NSET], obq[NSET], op[NSET], obp[NSET];
    for (int s = 0; s < nset; s++) {
        long long b = base + s * stride;
        oq[s]  = b + h + h;
        obq[s] = oq[s] + 3LL * h * h;
        op[s]  = obq[s] + 3 * h;
        obp[s] = op[s] + 1LL * h * h;
        wq[s] = (float*)malloc((size_t)h * 3 * h * 4);
        wp[s] = (float*)malloc((size_t)h * h * 4);
        bq[s] = (float*)malloc((size_t)3 * h * 4);
        bp[s] = (float*)malloc((size_t)h * 4);
        if (!wq[s] || !wp[s] || !bq[s] || !bp[s]) { printf("alloc w fail\n"); return 1; }
        memset(bq[s], 0, (size_t)3 * h * 4);   /* bias phai ZERO, khong duoc la rac */
        memset(bp[s], 0, (size_t)h * 4);
    }

    float* x    = (float*)malloc((size_t)n * h * 4);
    float* gout = (float*)malloc((size_t)n * h * 4);
    float* dx   = (float*)malloc((size_t)n * h * 4);
    float* ctx  = (float*)malloc((size_t)n * h * 4);
    float* gflat = (float*)calloc((size_t)npar, 4);
    if (!x || !gout || !dx || !ctx || !gflat) { printf("alloc fail\n"); return 1; }

    srand(12345);
    for (long long i = 0; i < (long long)n * h; i++) {
        x[i]    = ((float)rand() / RAND_MAX - 0.5f) * 0.8f;
        gout[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.8f;
    }
    for (int s = 0; s < nset; s++) {
        for (long long i = 0; i < (long long)h * 3 * h; i++)
            wq[s][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.03f;
        for (long long i = 0; i < (long long)h * h; i++)
            wp[s][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.03f;
    }

    HMODULE hm = LoadLibraryA("probe_attn_save.dll");
    if (!hm) { printf("load probe_attn_save.dll fail (err %lu)\n", GetLastError()); return 1; }
    fwdone_t  fwdone  = (fwdone_t) GetProcAddress(hm, "ProbeFwdBwdOne");
    fwdsave_t fwdsave = (fwdsave_t)GetProcAddress(hm, "ProbeFwdSave");
    bwdsaved_t bwdsav = (bwdsaved_t)GetProcAddress(hm, "ProbeBwdSaved");
    nodes_t   nodes   = (nodes_t)  GetProcAddress(hm, "ProbeGraphNodes");

    /* --- goi THANG tkv_simd_native.dll (bo qua wrapper managed cua .tkv) --- */
    HMODULE hs = LoadLibraryA("tkv_simd_native.dll");
    atnb_t atnb = NULL; sdpa_t sdpa = NULL;
    if (hs) {
        atnb = (atnb_t)GetProcAddress(hs, "TkvTorchAttentionBwd");
        sdpa = (sdpa_t)GetProcAddress(hs, "TkvTorchSdpa");
    }
    printf("tkv_simd_native: atnb=%p sdpa=%p\n", (void*)atnb, (void*)sdpa);
    if (!fwdone || !fwdsave || !bwdsav) { printf("sym fail\n"); return 1; }

    double wmb = (double)nset * (h * 3 * h + h * h) * 4 / 1048576.0;
    printf("n=%d h=%d nh=%d hd=%d reps=%d  %d bo weight = %.1f MB\n",
           n, h, nh, hd, reps, nset, wmb);

    /* ---------- warm ---------- */
    double f0 = 0, b0 = 0;
    int rc;
    for (int s = 0; s < nset; s++) {
        rc = fwdone(x, gout, wq[s], bq[s], wp[s], bp[s], dx, gflat, n, h, nh, hd,
                    oq[s], obq[s], op[s], obp[s], &f0, &b0);
        if (rc != 0) { printf("warm A rc=%d\n", rc); return 1; }
        rc = fwdsave(x, wq[s], bq[s], wp[s], bp[s], ctx, n, h, nh, hd, &f0);
        if (rc != 0) { printf("warm B rc=%d\n", rc); return 1; }
        rc = bwdsav(gout, dx, gflat, n, h, oq[s], obq[s], op[s], obp[s], 0, &b0);
        if (rc != 0) { printf("warm C rc=%d\n", rc); return 1; }
    }

    /* ---------- A) duong production hien tai: fwd+bwd trong 1 lenh ---------- */
    double bestA = 1e30, bestAf = 1e30, bestAb = 1e30;
    for (int r = 0; r < reps; r++) {
        double tot = 0, tf = 0, tb = 0;
        for (int s = 0; s < nset; s++) {
            double mf = 0, mb = 0;
            rc = fwdone(x, gout, wq[s], bq[s], wp[s], bp[s], dx, gflat, n, h, nh, hd,
                        oq[s], obq[s], op[s], obp[s], &mf, &mb);
            if (rc != 0) { printf("A rc=%d\n", rc); return 1; }
            tf += mf; tb += mb;
        }
        tot = tf + tb;
        if (tot < bestA) { bestA = tot; bestAf = tf / nset; bestAb = tb / nset; }
    }

    /* ---------- B+C) fwd giu graph + bwd dung lai ---------- */
    double bestBC = 1e30, bestB = 1e30, bestC = 1e30;
    for (int r = 0; r < reps; r++) {
        double tf = 0, tb = 0;
        for (int s = 0; s < nset; s++) {
            double mf = 0, mb = 0;
            rc = fwdsave(x, wq[s], bq[s], wp[s], bp[s], ctx, n, h, nh, hd, &mf);
            if (rc != 0) { printf("B rc=%d\n", rc); return 1; }
            rc = bwdsav(gout, dx, gflat, n, h, oq[s], obq[s], op[s], obp[s], 0, &mb);
            if (rc != 0) { printf("C rc=%d\n", rc); return 1; }
            tf += mf; tb += mb;
        }
        if (tf + tb < bestBC) { bestBC = tf + tb; bestB = tf / nset; bestC = tb / nset; }
    }

    printf("\n--- ms/layer, best of %d ---\n", reps);
    printf("A) TkvTorchAttentionBwd (fwd+bwd 1 lenh)  fwd %6.2f  bwd %6.2f  TONG %6.2f\n",
           bestAf, bestAb, bestA / nset);
    printf("B) fwd giu graph                          fwd %6.2f\n", bestB);
    printf("C) bwd dung graph da giu                  bwd %6.2f\n", bestC);
    printf("B+C                                       TONG %6.2f\n", bestBC / nset);
    printf("\n--- so sanh CAP MODEL (fwd production rieng + bwd) ---\n");
    printf("hien tai : fwd(ATen sdpa, khong graph) + A_bwd\n");
    printf("de xuat  : B + C\n");
    printf("A_bwd - C = %.2f ms/layer  <- phan fwd bi tinh LAI trong bwd\n",
           bestAb - bestC);
    printf("B - A_fwd = %.2f ms/layer  <- gia cua viec giu graph o fwd\n",
           bestB - bestAf);

    /* ---------- parity: dx + gflat phai khop bit ---------- */
    long long npar_s = obp[0] + h + 1024;
    float* dxA = (float*)malloc((size_t)n * h * 4);
    float* gA  = (float*)calloc((size_t)npar_s, 4);
    float* dxC = (float*)malloc((size_t)n * h * 4);
    float* gC  = (float*)calloc((size_t)npar_s, 4);
    if (!dxA || !gA || !dxC || !gC) { printf("parity alloc fail\n"); return 1; }
    rc = fwdone(x, gout, wq[0], bq[0], wp[0], bp[0], dxA, gA, n, h, nh, hd,
                oq[0], obq[0], op[0], obp[0], &f0, &b0);
    if (rc != 0) { printf("parity A rc=%d\n", rc); return 1; }
    rc = fwdsave(x, wq[0], bq[0], wp[0], bp[0], ctx, n, h, nh, hd, &f0);
    if (rc != 0) { printf("parity B rc=%d\n", rc); return 1; }
    rc = bwdsav(gout, dxC, gC, n, h, oq[0], obq[0], op[0], obp[0], 0, &b0);
    if (rc != 0) { printf("parity C rc=%d\n", rc); return 1; }

    double mdx = 0, mg = 0, sdx = 0, sg = 0;
    for (long long i = 0; i < (long long)n * h; i++) {
        double d = fabs((double)dxA[i] - dxC[i]);
        if (d > mdx) mdx = d;
        sdx += fabs((double)dxA[i]);
    }
    for (long long i = 0; i < npar_s; i++) {
        double d = fabs((double)gA[i] - gC[i]);
        if (d > mg) mg = d;
        sg += fabs((double)gA[i]);
    }
    /* ---------- D) goi THANG DLL, khong qua wrapper managed ---------- */
    if (atnb) {
        double bestD = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            for (int s = 0; s < nset; s++)
                atnb(x, gout, wq[s], bq[s], wp[s], bp[s], dx, gflat, n, h, nh, hd,
                     oq[s], obq[s], op[s], obp[s]);
            double t = now_ms() - t0;
            if (t < bestD) bestD = t;
        }
        printf("\nD) TkvTorchAttentionBwd goi THANG DLL   TONG %6.2f ms/layer\n", bestD / nset);
        printf("   A (qua probe .dll) - D = %.2f ms/layer  <- overhead goi\n",
               bestA / nset - bestD / nset);
    }
    if (sdpa) {
        double bestE = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            for (int s = 0; s < nset; s++) sdpa((const float*)gflat, ctx, n, nh, hd);
            double t = now_ms() - t0;
            if (t < bestE) bestE = t;
        }
        printf("E) TkvTorchSdpa goi THANG DLL (fwd)     %6.2f ms/layer\n", bestE / nset);
    }

    printf("\n--- chan doan graph ---\n");
    if (nodes) printf("node trong graph sau fwd = %d\n", nodes());
    printf("\n--- parity A vs B+C ---\n");
    printf("dx[0..2]  A = %.6e %.6e %.6e\n", dxA[0], dxA[1], dxA[2]);
    printf("dx[0..2]  C = %.6e %.6e %.6e\n", dxC[0], dxC[1], dxC[2]);
    printf("max_abs dx    = %.6e   (|dx| tong = %.6f)\n", mdx, sdx);
    printf("max_abs gflat = %.6e   (|gflat| tong = %.6f)\n", mg, sg);
    printf("%s\n", (mdx == 0.0 && mg == 0.0) ? "BIT-IDENTICAL" : "KHAC BIT");
    return 0;
}
