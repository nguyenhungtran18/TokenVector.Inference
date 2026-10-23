/* tkv_simd_native_m1.c — M1 drop-in shim cho tkv_simd_native.dll
 * (roadmap §14.3/§14.4 M1: "Goi BLAS san co qua dlopen — thay implementation
 * trong simd_gemm, giu nguyen ABI, .tkv khong doi 1 dong").
 *
 * ABI quan sat tu IL sinh ra (out/*.il, pinvokeimpl):
 *   void SgemmF32(float* a, float* b, float* o, int32 n);  // o = a x b, vuong n x n,
 *   void SgemmF64(double* a, double* b, double* o, int32 n); // row-major, beta=0
 * Chi 2 export nay duoc P/Invoke boi exe .tkv (cac simd_* elementwise con lai
 * la managed IL System.Numerics.Vector, KHONG goi DLL nay).
 *
 * Than GEMM forward sang OpenBLAS cblas_sgemm/dgemm, nap runtime (LoadLibrary)
 * tu "m1blas.dll" (build script copy OpenBLAS cua scipy vao cung thu muc exe)
 * hoac duong tuyet doi trong env TKV_M1_BLAS. So thread: env TKV_M1_THREADS
 * (mac dinh 14 — khop moi trong do torch 14T). Neu thieu BLAS -> fallback
 * naive + canh bao re (gate van chay duoc, cham).
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111

typedef void (*cblas_sgemm_t)(int, int, int, int, int, int,
                              float, const float*, int, const float*, int,
                              float, float*, int);
typedef void (*cblas_dgemm_t)(int, int, int, int, int, int,
                              double, const double*, int, const double*, int,
                              double, double*, int);
typedef void (*oblas_set_t)(int);

static cblas_sgemm_t p_sgemm = NULL;
static cblas_dgemm_t p_dgemm = NULL;
static int g_ready = 0;
static int g_fallback = 0;

static HMODULE load_blas(void) {
    char path[MAX_PATH * 2];
    const char* env = getenv("TKV_M1_BLAS");
    HMODULE h = NULL;
    if (env && env[0]) {
        h = LoadLibraryA(env);
        if (h) return h;
        fprintf(stderr, "[m1] TKV_M1_BLAS khong nap duoc: %s\n", env);
    }
    h = LoadLibraryA("m1blas.dll");   /* exe dir nam trong default search */
    if (h) return h;
    return NULL;
}

static FARPROC sym(HMODULE h, const char* a, const char* b) {
    FARPROC p = GetProcAddress(h, a);
    if (!p && b) p = GetProcAddress(h, b);
    return p;
}

static void init_once(void) {
    HMODULE h;
    oblas_set_t setn;
    int nthr = 14;
    const char* env;
    if (g_ready) return;
    h = load_blas();
    if (h) {
        p_sgemm = (cblas_sgemm_t)sym(h, "scipy_cblas_sgemm", "cblas_sgemm");
        p_dgemm = (cblas_dgemm_t)sym(h, "scipy_cblas_dgemm", "cblas_dgemm");
        setn = (oblas_set_t)sym(h, "scipy_openblas_set_num_threads",
                                   "openblas_set_num_threads");
        env = getenv("TKV_M1_THREADS");
        if (env && env[0]) nthr = atoi(env);
        if (nthr < 1) nthr = 1;
        if (setn) setn(nthr);
    }
    if (!p_sgemm) {
        g_fallback = 1;
        fprintf(stderr, "[m1] KHONG tim thay cblas_sgemm -> fallback naive "
                        "(thread=%d).\n", nthr);
    } else {
        fprintf(stderr, "[m1] OpenBLAS da nap, cblas_sgemm OK, threads=%d%s\n",
                nthr, p_dgemm ? "" : " (chi co sgemm)");
    }
    g_ready = 1;
}

static void naive_sgemm(const float* a, const float* b, float* o, int n) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (int p = 0; p < n; p++) o[(size_t)i * n + p] = 0.0f;
        for (k = 0; k < n; k++) {
            float av = a[(size_t)i * n + k];
            const float* brow = b + (size_t)k * n;
            for (j = 0; j < n; j++) o[(size_t)i * n + j] += av * brow[j];
        }
    }
}

__declspec(dllexport) void __stdcall SgemmF32(float* a, float* b, float* o, int n) {
    if (!g_ready) init_once();
    if (p_sgemm) {
        p_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                n, n, n, 1.0f, a, n, b, n, 0.0f, o, n);
    } else {
        naive_sgemm(a, b, o, n);
    }
}

__declspec(dllexport) void __stdcall SgemmF64(double* a, double* b, double* o, int n) {
    if (!g_ready) init_once();
    if (p_dgemm) {
        p_dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                n, n, n, 1.0, a, n, b, n, 0.0, o, n);
    } else {
        int i, j, k;
        for (i = 0; i < n; i++) {
            for (int p = 0; p < n; p++) o[(size_t)i * n + p] = 0.0;
            for (k = 0; k < n; k++) {
                double av = a[(size_t)i * n + k];
                const double* brow = b + (size_t)k * n;
                for (j = 0; j < n; j++) o[(size_t)i * n + j] += av * brow[j];
            }
        }
    }
}

BOOL WINAPI DllMain(HMODULE m, DWORD r, void* p) {
    (void)m; (void)r; (void)p;
    return TRUE;
}
