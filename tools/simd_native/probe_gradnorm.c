/* probe_gradnorm.c - F32GradNorm native vs vong interpreted .tkv
 *
 * Boi canh: _norm_groups (trainer.tkv) tinh grad_norm bang 12 worker
 * INTERPRETED, do duoc 422-438 ms cho 124.4M float (498 MB doc).
 * Cung khoi luong do f32_copy_to_buf (memcpy native) chi 78 ms.
 * => 5.4x chenh lech la chi phi thong dich, khong phai bang thong.
 *
 * DLL da export F32GradNorm (tkv_simd_native.c L2014) nhung KHONG noi nao
 * trong .tkv goi -> dead code. Probe nay do no de biet gia tri neu bind.
 *
 * HOP DONG BIT-EXACT (comment L2005-2013 cua tkv_simd_native.c):
 *   - binh phuong tren f32: float p = g[i]*g[i]
 *   - tich luy f64 TUAN TU i = 0..n-1, khong gom, khong song song
 *   - sqrt tren f64
 * => cung thu tu voi vong interpreted -> ket qua TUNG BIT giong nhau.
 * Probe verify dieu do bang cach tu tinh lai y het trong C.
 *
 * Build: powershell -ExecutionPolicy Bypass -File build_probe_gradnorm.ps1
 * Chay : ./probe_gradnorm.exe   (can tkv_simd_native.dll cung thu muc)
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <time.h>

typedef double (*gn_t)(const float*, int, int);

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 124439808;
    printf("=== F32GradNorm n=%d (%.1f MB doc) ===\n\n", n, n * 4.0 / (1024.0 * 1024.0));

    float* g = (float*)malloc((size_t)n * 4);
    if (!g) { printf("malloc fail\n"); return 1; }
    /* gia tri giong _opt_split.tkv: g[i] = 0.001 -> norm = 0.001*sqrt(n) */
    for (int i = 0; i < n; i++) g[i] = 0.001f;

    HMODULE h = LoadLibraryA("tkv_simd_native.dll");
    if (!h) { printf("load fail\n"); return 1; }
    gn_t gn = (gn_t)GetProcAddress(h, "F32GradNorm");
    if (!gn) { printf("sym fail: F32GradNorm\n"); return 1; }

    /* ---- reference: vong C tuong duong vong interpreted .tkv ---- */
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        float p = g[i] * g[i];
        acc += (double)p;
    }
    double ref = sqrt(acc);

    /* ---- native ---- */
    double best = 1e30, val = 0.0;
    for (int r = 0; r < 5; r++) {
        double t0 = now_ms();
        val = gn(g, n, n);
        double t1 = now_ms();
        if (t1 - t0 < best) best = t1 - t0;
    }

    printf("reference (vong C tuan tu f64) : %.10f\n", ref);
    printf("F32GradNorm native             : %.10f\n", val);
    printf("BIT-IDENTICAL                  : %s\n\n", (ref == val) ? "YES (0.0)" : "NO");

    printf("native best-of-5 : %8.2f ms\n", best);
    printf("bang thong       : %8.2f GB/s\n", (n * 4.0) / (best / 1000.0) / 1e9);
    printf("\nSo sanh (tu probe .tkv cung may):\n");
    printf("  _norm_groups 12 worker interpreted : 422-438 ms\n");
    printf("  f32_copy_to_buf cung 498 MB        :  78 ms\n");
    if (best > 0) {
        printf("  => native nhanh hon interpreted    : %.2fx\n", 430.0 / best);
        printf("  => tiet kiem moi buoc              : %.0f ms\n", 430.0 - best);
    }
    return 0;
}
