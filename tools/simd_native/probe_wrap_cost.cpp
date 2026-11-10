/* probe_wrap_cost.cpp - PROBE. Do chi phi wrapper managed theo KICH THUOC list.
 *
 * Gia thuyet can kiem (s0.144 muc 3): wrapper .tkv ton ~7 ms/call tren list
 * 475 MB, tuc chi phi TI LE voi kich thuoc list.
 *
 * Cach kiem: mot ham no-op chi doc phan tu dau. Neu chi phi tang theo kich
 * thuoc list thi gia thuyet dung; neu phang thi gia thuyet SAI.
 *
 * Build: build_probe_wrap_cost.ps1 -> probe_wrap_cost.dll
 */
#include <cstdio>
#include <cstdint>

#define PROBE_EXPORT extern "C" __declspec(dllexport)

/* Doc 1 phan tu, khong lam gi khac. */
PROBE_EXPORT int ProbeTouch(const float* g, int64_t len) {
    if (!g || len <= 0) return -1;
    volatile float v = g[0] + g[len - 1];
    (void)v;
    return 0;
}

/* Ghi 1 phan tu — buoc phai that su cham vao bo nho. */
PROBE_EXPORT int ProbeTouchW(float* g, int64_t len) {
    if (!g || len <= 0) return -1;
    g[0] = g[0] + 1.0f;
    return 0;
}
