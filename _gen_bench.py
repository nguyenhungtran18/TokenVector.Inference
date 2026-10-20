# -*- coding: utf-8 -*-
"""One-off: generate tv/tests/_bench_p53.tkv — A/B/C timing for P5.2/P5.3.

A = HEAD's gpt2_attention / gpt2_attention_bwd with simd=1 (P5.1 only:
    proj/dctx/dw_proj GEMM, core attention still scalar)
B = current code with simd=1 (P5.1 + P5.2 score/ctx GEMM + P5.3 dp/ds/dqkv GEMM)
C = current code with simd=0 (fully scalar reference)

Same process, same weights, same input => single-variable comparison.
The two HEAD functions are extracted verbatim (only the `def` name changes)
so A is a faithful copy of the committed P5.1 baseline.
"""
import io
import subprocess

OUT = "tv/tests/_bench_p53.tkv"

# PIN commit chua P5.1 truoc P5.2/P5.3. SAU KHI COMMIT, `HEAD` se chua ma moi
# -> A se GIONG B va benchmark mat y nghia (B/A = 1.0 vo nghia).
# Khong doi hash nay; neu phai tinh lai thi chon lai commit truoc P5.2/P5.3.
BASELINE_COMMIT = "4a14acc"

HEAD_SRC = subprocess.check_output(
    ["git", "show", BASELINE_COMMIT + ":tv/training/gpt2_ref.tkv"]
).decode("utf-8-sig")


def grab(text, start_marker, end_marker):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[i:j]


fwd = grab(HEAD_SRC, 'def gpt2_attention(x: "Tensor"', "# ---- MLP: linear -> gelu")
bwd = grab(HEAD_SRC, 'def gpt2_attention_bwd(x: "Tensor"', "# [dx, dw1, db1, dw2, db2]")
fwd = fwd.replace("def gpt2_attention(x:", "def attn_a(x:", 1)
bwd = bwd.replace("def gpt2_attention_bwd(x:", "def attnb_a(x:", 1)

HEADER = '''# BENCH P53 — A/B/C attention 124M: P5.1 vs P5.1+P5.2+P5.3 vs scalar
# ----------------------------------------------------------------------------
# A = attn_a / attnb_a     : ban HEAD (chi P5.1)            , simd = 1
# B = gpt2_attention(_bwd) : ban hien tai (P5.1 + P5.2/P5.3), simd = 1
# C = gpt2_attention(_bwd) : duong scalar                    , simd = 0
#
# Cung process, cung trong so, cung dau vao -> moi bien chinh la P5.2/P5.3.
# A duoc chen NGUYEN VANC tu commit BASELINE_COMMIT = 4a14acc (chi doi ten
# def) -> chan sai so copy. Doi bang cach chay lai _gen_bench.py (co pin SHA; KHONG
# dung `HEAD` vi sau khi commit P5.3 thi HEAD = ma moi).
# Timing dat toan bo trong main (bug compiler #8: now_ms() trong local ham
# doc SAI) — xem docs/perf-plan-simd-vs-pytorch.md.
#
# Nguon seq: tv/tests/_seq_cfg.txt dong 1.
#   printf "1024\\n0\\n1\\n" > tv/tests/_seq_cfg.txt
#   tkvc tv/tests/_bench_p53.tkv out/b53.exe main && ./out/b53.exe
import tv.training.gpt2_ref124
import tv.training.simd_matmul

def _is_comment(ln: 'str') -> 'i32':
    if len(ln) < 1:
        return 1
    if ln[0] == "#":
        return 1
    return 0

def _read_n(path: 'str') -> 'i64':
    txt = read_file(path)
    lines = txt.split("\\n")
    ln = lines[0]
    if _is_comment(ln) == 1:
        return 8
    toks = ln.split(" ")
    for j in range(len(toks)):
        if len(toks[j]) > 0:
            return int(float(toks[j]))
    return 8

# now_ms() co do phan giai ~15.6 ms (Windows) -> o seq8 mot lan do vo nghia
# (perf-plan, bai hoc 1). Dong 4 = so lan lap de TRUNG BINH.
def _read_reps(path: 'str') -> 'i64':
    txt = read_file(path)
    lines = txt.split("\\n")
    if len(lines) < 4:
        return 1
    ln = lines[3]
    if _is_comment(ln) == 1:
        return 1
    toks = ln.split(" ")
    for j in range(len(toks)):
        if len(toks[j]) > 0:
            r = int(float(toks[j]))
            if r < 1:
                return 1
            return r
    return 1

def _mdiff(a: "Tensor", b: "Tensor") -> 'f32':
    da = a.data
    db = b.data
    n = len(da)
    if len(db) < n:
        n = len(db)
    m = 0.0
    for i in range(n):
        d = da[i] - db[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m

'''

BODY = '''
def main() -> 'i32':
    n = _read_n("tv/tests/_seq_cfg.txt")
    reps = _read_reps("tv/tests/_seq_cfg.txt")
    h = HiddenSize
    hd = HeadDim
    print("BENCH P53 seq n=" + str(n) + " h=" + str(h) + " hd=" + str(hd)
          + " NHeads=" + str(NHeads) + " reps=" + str(reps))

    # ---- trong so + dau vao, sinh dinh dang (khong gpt2_init -> nhe) ------
    w_qkv = t_zeros([h, 3 * h])
    b_qkv = t_zeros([3 * h])
    w_proj = t_zeros([h, h])
    b_proj = t_zeros([h])
    for i in range(h * 3 * h):
        w_qkv.data[i] = ((float((i * 13 + 7) % 97) - 48.0) / 48.0) * 0.02
    for i in range(3 * h):
        b_qkv.data[i] = ((float((i * 5 + 3) % 53) - 26.0) / 26.0) * 0.01
    for i in range(h * h):
        w_proj.data[i] = ((float((i * 11 + 5) % 89) - 44.0) / 44.0) * 0.02
    for i in range(h):
        b_proj.data[i] = ((float((i * 3 + 1) % 47) - 23.0) / 23.0) * 0.01
    x = t_zeros([n, h])
    for i in range(n * h):
        x.data[i] = ((float((i * 13 + 7) % 97) - 48.0) / 48.0) * 0.02
    gout = t_zeros([n, h])
    for i in range(n * h):
        gout.data[i] = 0.1

    # ---- A: P5.1 only (ban HEAD), warmup 1 + do 1 ------------------------
    wA = attn_a(x, w_qkv, b_qkv, w_proj, b_proj, n, 1)
    wAb = attnb_a(x, w_qkv, b_qkv, w_proj, b_proj, n, gout, 1)
    t0 = now_ms()
    for rep in range(reps):
        oA = attn_a(x, w_qkv, b_qkv, w_proj, b_proj, n, 1)
    t1 = now_ms()
    t0b = now_ms()
    for rep in range(reps):
        rA = attnb_a(x, w_qkv, b_qkv, w_proj, b_proj, n, gout, 1)
    t1b = now_ms()
    dt_Af = (t1 - t0) / float(reps)
    dt_Ab = (t1b - t0b) / float(reps)

    # ---- B: hien tai P5.1+P5.2+P5.3, simd=1, warmup 1 + do 1 ------------
    wB = gpt2_attention(x, w_qkv, b_qkv, w_proj, b_proj, n, 1)
    wBb = gpt2_attention_bwd(x, w_qkv, b_qkv, w_proj, b_proj, n, gout, 1)
    t0 = now_ms()
    for rep in range(reps):
        oB = gpt2_attention(x, w_qkv, b_qkv, w_proj, b_proj, n, 1)
    t1 = now_ms()
    t0b = now_ms()
    for rep in range(reps):
        rB = gpt2_attention_bwd(x, w_qkv, b_qkv, w_proj, b_proj, n, gout, 1)
    t1b = now_ms()
    dt_Bf = (t1 - t0) / float(reps)
    dt_Bb = (t1b - t0b) / float(reps)

    # ---- C: scalar, khong warmup rieng (process da am) --------------------
    t0 = now_ms()
    for rep in range(reps):
        oC = gpt2_attention(x, w_qkv, b_qkv, w_proj, b_proj, n, 0)
    t1 = now_ms()
    t0b = now_ms()
    for rep in range(reps):
        rC = gpt2_attention_bwd(x, w_qkv, b_qkv, w_proj, b_proj, n, gout, 0)
    t1b = now_ms()
    dt_Cf = (t1 - t0) / float(reps)
    dt_Cb = (t1b - t0b) / float(reps)

    # ---- self-check: A, B, C phai tinh cung mot ket qua (SIMD lech ULP) ---
    print("BENCH selfcheck fwd  |A-B|=" + str(_mdiff(oA, oB))
          + "  |B-C|=" + str(_mdiff(oB, oC)))
    dA = rA[0]
    dB = rB[0]
    dC = rC[0]
    print("BENCH selfcheck bwd dx |A-B|=" + str(_mdiff(dA, dB))
          + "  |B-C|=" + str(_mdiff(dB, dC)))
    wA2 = rA[1]
    wB2 = rB[1]
    print("BENCH selfcheck bwd dw_qkv |A-B|=" + str(_mdiff(wA2, wB2)))

    # ---- bao cao ----------------------------------------------------------
    print("BENCH A P5.1-only simd : fwd=" + str(dt_Af) + " ms  bwd=" + str(dt_Ab)
          + " ms  tong=" + str(dt_Af + dt_Ab) + " ms")
    print("BENCH B P5.1+P5.2+P5.3 : fwd=" + str(dt_Bf) + " ms  bwd=" + str(dt_Bb)
          + " ms  tong=" + str(dt_Bf + dt_Bb) + " ms")
    print("BENCH C scalar          : fwd=" + str(dt_Cf) + " ms  bwd=" + str(dt_Cb)
          + " ms  tong=" + str(dt_Cf + dt_Cb) + " ms")
    if dt_Bf + dt_Bb > 0.0:
        print("BENCH speedup B vs A (P5.2+P5.3) = "
              + str((dt_Af + dt_Ab) / (dt_Bf + dt_Bb)) + "x")
    if dt_Cf + dt_Cb > 0.0:
        print("BENCH speedup B vs C (SIMD vs scalar) = "
              + str((dt_Cf + dt_Cb) / (dt_Bf + dt_Bb)) + "x")
    print("BENCH done")
    return 0
'''

with io.open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    f.write(HEADER)
    f.write(fwd.rstrip("\n") + "\n\n")
    f.write(bwd.rstrip("\n") + "\n\n")
    f.write(BODY)

print("wrote", OUT)
print("  attn_a lines:", fwd.count("\n"), " attnb_a lines:", bwd.count("\n"))
