/* probe_attn_save.cpp - PROBE (khong phai production).
 *
 * Cau hoi: "luu ctx+lse o fwd" tiet kiem duoc bao nhieu trong
 * TkvTorchAttentionBwd (hien tai 64-88 ms/layer)?
 *
 * Ba cach do:
 *   A) ProbeFwdBwdOne  - y het TkvTorchAttentionBwd, nhung tach timer fwd/bwd.
 *   B) ProbeFwdSave    - dung graph ATen, GIU graph + tensor (khong copy ra).
 *   C) ProbeBwdSaved   - backward tren graph da giu -> khong recompute fwd.
 *
 * Neu B+C < A thi "giu graph" la duong dung, va khong can tu viet cong thuc
 * attention backward (giu duoc BIT-IDENTICAL voi autograd).
 *
 * Build: build_probe_attn_save.ps1 -> probe_attn_save.dll
 */
#include <torch/torch.h>
#include <chrono>
#include <cstring>
#include <functional>

#define PROBE_EXPORT extern "C" __declspec(dllexport)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               steady_clock::now().time_since_epoch()).count();
}

/* ---- graph giu lai giua fwd va bwd (probe: 1 slot, single-thread) ---- */
struct SavedGraph {
    torch::Tensor x, wq, bq, wp, bp;   /* la */
    torch::Tensor proj;                /* dinh graph */
    bool valid = false;
};
static SavedGraph g_saved;

static void zero_grads(SavedGraph& s) {
    if (s.x.grad().defined())  s.x.grad().zero_();
    if (s.wq.grad().defined()) s.wq.grad().zero_();
    if (s.bq.grad().defined()) s.bq.grad().zero_();
    if (s.wp.grad().defined()) s.wp.grad().zero_();
    if (s.bp.grad().defined()) s.bp.grad().zero_();
}

/* ================= A) y het production, tach timer ================= */
PROBE_EXPORT int ProbeFwdBwdOne(
        const float* x, const float* gout,
        const float* w_qkv, const float* b_qkv,
        const float* w_proj, const float* b_proj,
        float* dx, float* gflat,
        int64_t n, int64_t h, int64_t nheads, int64_t hd,
        long long o_qkv, long long o_bqkv, long long o_proj, long long o_bproj,
        double* ms_fwd, double* ms_bwd) {
    if (!x || !gout || !w_qkv || !b_qkv || !w_proj || !b_proj || !dx) return -1;
    try {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        const int64_t T = n;
        if (h != nheads * hd) return -2;

        double t0 = now_ms();
        auto t_x  = torch::from_blob(const_cast<float*>(x),   {T, h},   opt.requires_grad(true));
        auto t_wq = torch::from_blob(const_cast<float*>(w_qkv), {h, 3 * h}, opt.requires_grad(true));
        auto t_bq = torch::from_blob(const_cast<float*>(b_qkv), {3 * h},   opt.requires_grad(true));
        auto t_wp = torch::from_blob(const_cast<float*>(w_proj), {h, h},  opt.requires_grad(true));
        auto t_bp = torch::from_blob(const_cast<float*>(b_proj), {h},     opt.requires_grad(true));

        auto t_qkv = at::addmm(t_bq, t_x, t_wq);
        auto t_q3  = t_qkv.view({T, 3, nheads, hd});
        auto q = t_q3.select(1, 0).transpose(0, 1).unsqueeze(0);
        auto k = t_q3.select(1, 1).transpose(0, 1).unsqueeze(0);
        auto v = t_q3.select(1, 2).transpose(0, 1).unsqueeze(0);
        auto t_attn = at::scaled_dot_product_attention(q, k, v, std::nullopt,
                                                       0.0, true, std::nullopt);
        auto t_ctx  = t_attn.transpose(1, 2).reshape({T, h}).contiguous();
        auto t_proj = at::addmm(t_bp, t_ctx, t_wp);
        double t1 = now_ms();

        auto t_gy = torch::from_blob(const_cast<float*>(gout), {T, h}, opt);
        t_proj.backward(t_gy);
        double t2 = now_ms();

        if (ms_fwd) *ms_fwd = t1 - t0;
        if (ms_bwd) *ms_bwd = t2 - t1;

        memcpy(dx, t_x.grad().const_data_ptr<float>(), (size_t)T * h * sizeof(float));
        if (gflat && o_qkv  >= 0) memcpy(gflat + o_qkv,  t_wq.grad().const_data_ptr<float>(), (size_t)h * 3 * h * sizeof(float));
        if (gflat && o_bqkv >= 0) memcpy(gflat + o_bqkv, t_bq.grad().const_data_ptr<float>(), (size_t)3 * h * sizeof(float));
        if (gflat && o_proj >= 0) memcpy(gflat + o_proj, t_wp.grad().const_data_ptr<float>(), (size_t)h * h * sizeof(float));
        if (gflat && o_bproj >= 0) memcpy(gflat + o_bproj, t_bp.grad().const_data_ptr<float>(), (size_t)h * sizeof(float));
        return 0;
    } catch (const std::exception&) { return -4; } catch (...) { return -5; }
}

/* ================= B) fwd: dung graph, GIU lai ================= */
PROBE_EXPORT int ProbeFwdSave(
        const float* x, const float* w_qkv, const float* b_qkv,
        const float* w_proj, const float* b_proj,
        float* ctx_out,
        int64_t n, int64_t h, int64_t nheads, int64_t hd,
        double* ms_fwd) {
    if (!x || !w_qkv || !b_qkv || !w_proj || !b_proj || !ctx_out) return -1;
    try {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        const int64_t T = n;
        if (h != nheads * hd) return -2;

        double t0 = now_ms();
        /* Chi clone x: no la input production, phai song qua lan goi.
         * w/b la tham so, caller giu bo nho on dinh ca step -> from_blob thang,
         * dung nhu TkvTorchAttentionBwd dang lam (clone ca 5 ton 6.6 ms). */
        auto t_x  = torch::from_blob(const_cast<float*>(x),   {T, h},   opt).clone().set_requires_grad(true);
        auto t_wq = torch::from_blob(const_cast<float*>(w_qkv), {h, 3 * h}, opt).requires_grad_(true);
        auto t_bq = torch::from_blob(const_cast<float*>(b_qkv), {3 * h},   opt).requires_grad_(true);
        auto t_wp = torch::from_blob(const_cast<float*>(w_proj), {h, h},  opt).requires_grad_(true);
        auto t_bp = torch::from_blob(const_cast<float*>(b_proj), {h},     opt).requires_grad_(true);

        auto t_qkv = at::addmm(t_bq, t_x, t_wq);
        auto t_q3  = t_qkv.view({T, 3, nheads, hd});
        auto q = t_q3.select(1, 0).transpose(0, 1).unsqueeze(0);
        auto k = t_q3.select(1, 1).transpose(0, 1).unsqueeze(0);
        auto v = t_q3.select(1, 2).transpose(0, 1).unsqueeze(0);
        auto t_attn = at::scaled_dot_product_attention(q, k, v, std::nullopt,
                                                       0.0, true, std::nullopt);
        auto t_ctx  = t_attn.transpose(1, 2).reshape({T, h}).contiguous();
        auto t_proj = at::addmm(t_bp, t_ctx, t_wp);

        memcpy(ctx_out, t_ctx.const_data_ptr<float>(), (size_t)T * h * sizeof(float));

        g_saved.x = t_x; g_saved.wq = t_wq; g_saved.bq = t_bq;
        g_saved.wp = t_wp; g_saved.bp = t_bp; g_saved.proj = t_proj;
        g_saved.valid = true;
        double t1 = now_ms();
        if (ms_fwd) *ms_fwd = t1 - t0;
        return 0;
    } catch (const std::exception&) { return -4; } catch (...) { return -5; }
}

/* ================= C) bwd: dung graph da giu, KHONG recompute ========= */
/* keep_graph: 1 = goi lai duoc nhieu lan (do lap), 0 = giai phong buffer
 * (dung nhu production, chi backward MOT lan). */
PROBE_EXPORT int ProbeBwdSaved(
        const float* gout, float* dx, float* gflat,
        int64_t n, int64_t h,
        long long o_qkv, long long o_bqkv, long long o_proj, long long o_bproj,
        int64_t keep_graph, double* ms_bwd) {
    if (!g_saved.valid) return -3;
    if (!gout || !dx) return -1;
    try {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        const int64_t T = n;
        double t0 = now_ms();
        zero_grads(g_saved);
        auto t_gy = torch::from_blob(const_cast<float*>(gout), {T, h}, opt);
        torch::autograd::backward({g_saved.proj}, {t_gy}, /*keep_graph=*/keep_graph != 0);
        double t1 = now_ms();
        if (ms_bwd) *ms_bwd = t1 - t0;

        memcpy(dx, g_saved.x.grad().const_data_ptr<float>(), (size_t)T * h * sizeof(float));
        if (gflat && o_qkv  >= 0) memcpy(gflat + o_qkv,  g_saved.wq.grad().const_data_ptr<float>(), (size_t)h * 3 * h * sizeof(float));
        if (gflat && o_bqkv >= 0) memcpy(gflat + o_bqkv, g_saved.bq.grad().const_data_ptr<float>(), (size_t)3 * h * sizeof(float));
        if (gflat && o_proj >= 0) memcpy(gflat + o_proj, g_saved.wp.grad().const_data_ptr<float>(), (size_t)h * h * sizeof(float));
        if (gflat && o_bproj >= 0) memcpy(gflat + o_bproj, g_saved.bp.grad().const_data_ptr<float>(), (size_t)h * sizeof(float));
        return 0;
    } catch (const std::exception&) { return -4; } catch (...) { return -5; }
}

PROBE_EXPORT int ProbeSavedValid() { return g_saved.valid ? 1 : 0; }

/* Chan doan: graph co con song khong, va no giu bao nhieu node?
 * -1 = chua co, 0 = grad_fn da bi giai phong, >0 = so node con lai. */
PROBE_EXPORT int ProbeGraphNodes() {
    if (!g_saved.valid) return -1;
    if (!g_saved.proj.defined()) return -2;
    auto g = g_saved.proj.grad_fn();
    if (!g) return 0;
    int cnt = 0;
    std::function<void(const c10::intrusive_ptr<torch::autograd::Node>&)> walk =
        [&](const c10::intrusive_ptr<torch::autograd::Node>& nd) {
            if (!nd || cnt > 64) return;
            cnt++;
            for (const auto& e : nd->next_edges()) walk(e.function);
        };
    walk(g);
    return cnt;
}
PROBE_EXPORT void ProbeSavedClear() { g_saved = SavedGraph(); }

/* ============ D) lse co lay duoc tu ATen khong, gia bao nhieu? ========= */
/* Tra ve lse [nheads, n] (log-sum-exp theo hang, truoc softmax) de doi chieu.
 * Dung de kiem tra: neu muon di duong native kernel thi lse phai co. */
PROBE_EXPORT int ProbeSdpaLse(
        const float* qkv, float* y, float* lse,
        int64_t n, int64_t nheads, int64_t hd, double* ms) {
    if (!qkv || !y || !lse) return -1;
    try {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        const int64_t h = nheads * hd;
        double t0 = now_ms();
        auto t_q = at::empty({1, nheads, n, hd}, opt);
        auto t_k = at::empty({1, nheads, n, hd}, opt);
        auto t_v = at::empty({1, nheads, n, hd}, opt);
        const float* qsrc = qkv;
        const float* ksrc = qkv + h;
        const float* vsrc = qkv + 2 * h;
        const int64_t stride = 3 * h;
        float* qp = t_q.data_ptr<float>();
        float* kp = t_k.data_ptr<float>();
        float* vp = t_v.data_ptr<float>();
        for (int64_t hi = 0; hi < nheads; hi++)
            for (int64_t t = 0; t < n; t++) {
                memcpy(qp + (hi * n + t) * hd, qsrc + t * stride + hi * hd, hd * sizeof(float));
                memcpy(kp + (hi * n + t) * hd, ksrc + t * stride + hi * hd, hd * sizeof(float));
                memcpy(vp + (hi * n + t) * hd, vsrc + t * stride + hi * hd, hd * sizeof(float));
            }
        /* math backend: S = QK^T*scale + causal mask, lse = logsumexp(S, -1) */
        auto scale = 1.0 / std::sqrt((double)hd);
        auto S = at::matmul(t_q, t_k.transpose(-1, -2)) * scale;   /* [1,nh,n,n] */
        auto mask = at::ones({n, n}, opt).tril().logical_not();
        S = S.masked_fill(mask, -std::numeric_limits<double>::infinity());
        auto l = at::logsumexp(S, -1);                              /* [1,nh,n] */
        auto P = at::softmax(S, -1);
        auto t_y = at::matmul(P, t_v);                              /* [1,nh,n,hd] */
        double t1 = now_ms();
        if (ms) *ms = t1 - t0;

        const float* yp = t_y.const_data_ptr<float>();
        for (int64_t hi = 0; hi < nheads; hi++)
            for (int64_t t = 0; t < n; t++)
                memcpy(y + t * h + hi * hd, yp + (hi * n + t) * hd, hd * sizeof(float));
        memcpy(lse, l.const_data_ptr<float>(), (size_t)nheads * n * sizeof(float));
        return 0;
    } catch (const std::exception&) { return -4; } catch (...) { return -5; }
}
