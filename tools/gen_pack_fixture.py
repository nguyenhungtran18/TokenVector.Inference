#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture PACKING + SHARD cho G3 (data pipeline).

Day la IMPLEMENTATION THAM CHIEU cua spec trong tv/data/dataset.tkv —
file nay va .tkv phai doi xung CHINH XAC tung buoc (cung hang so):

  Packing:  concat cac doc theo thu tu; moi doc ket thuc bang eos_id
            (eos THUOC doc do); cat thanh chunk T cu phay; tail KHONG duong
            -> BO (khong pad).
  Mask:     causal (j<=i) va block-diagonal (j<=i va doc[j]==doc[i]).
  Shard:    text file moi dong so token; stream = shard xep theo
            order(seed); sample = T token tiep tuyen; bo chunk cuoi.
  RNG:      LGC 31-bit: state' = (state*1103515245 + 12345) mod 2^31
            Fisher-Yates: i=n-1..1: j = state' mod (i+1); swap.
  Checksum: h = (h*31 + x) mod 2^31 tren chuoi token da gop.

Docs duoc tokenize bang chinh vocab/merges HF (tokenizers, OFFLINE) —
cung duong dan voi gen_tokenizer_fixture.py.

Chay:  python tools\\gen_pack_fixture.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "tv", "tests", "data")
OUT = os.path.join(ROOT, "tv", "tests", "fixtures", "pack_cases.txt")

SEQ_LEN = 32          # T cua packed sequence
EOS = 50256           # endoftext cua GPT-2
SHARD_SIZE = 16
SEED_A = 42
SEED_B = 43
SAMPLE_SEQ_LEN = 16
N_STREAM_SAMPLES_FIXTURE = 3   # so sample dau ghi vao fixture

LCG_A = 1103515245
LCG_C = 12345
LCG_M = 1 << 31

# 10 doan van — lay tu TEXTS cua gen_tokenizer_fixture.py (trung voi
# tok_cases.txt) de tokenizer da parity phu.
TEXTS = [
    "Hello, world! This is a GPT-2 tokenizer test.",
    "  leading and trailing spaces   ",
    "don't can't they're we've I'll you'd it's",
    "Newlines:\nand\ttabs, plus  punctuation!!! ??? 1234567890 3.14159",
    "Tôi là TokenVector, huấn luyện mô hình ngôn ngữ lớn. "
    "Xây dựng với ký tự đặc biệt: ư, ơ, đ, ạ!",
    "café naïve résumé 日本語 mixed 中文 text 한국어",
    "hi \U0001f44b there \U0001f600 emoji and lone ' quote",
    "Numbers 0123456789 ordinals 1st 2nd 3rd hex 0xFF IP 192.168.1.1",
    "The quick brown fox jumps over the lazy dog 1234567890.",
    "Việt Nam Hà Nội Sài Gòn — token vector inference engine.",
]


# ---- spec doi xung (giong y htkv) -----------------------------------------

def rng_step(state):
    return (state * LCG_A + LCG_C) % LCG_M


def shard_order(n, seed):
    state = seed % LCG_M
    xs = list(range(n))
    i = n - 1
    while i > 0:
        state = rng_step(state)
        j = state % (i + 1)
        xs[i], xs[j] = xs[j], xs[i]
        i = i - 1
    return xs


def pack_sequences(docs, seq_len, eos):
    seqs, docids = [], []
    cur, curd = [], []
    for k, d in enumerate(docs):
        for t in d:
            cur.append(t)
            curd.append(k)
            if len(cur) == seq_len:
                seqs.append(cur)
                docids.append(curd)
                cur, curd = [], []
        cur.append(eos)
        curd.append(k)
        if len(cur) == seq_len:
            seqs.append(cur)
            docids.append(curd)
            cur, curd = [], []
    return seqs, docids   # tail con lai bi BO


def block_causal_mask(doc_ids):
    t = len(doc_ids)
    return [[1 if (j <= i and doc_ids[j] == doc_ids[i]) else 0
             for j in range(t)] for i in range(t)]


def causal_mask(t):
    return [[1 if j <= i else 0 for j in range(t)] for i in range(t)]


def checksum(xs):
    h = 0
    for x in xs:
        h = (h * 31 + x) % LCG_M
    return h


def main():
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    except ImportError:
        print("CANH BAO: khong tim thay tokenizers.")
        sys.exit(2)

    tok = Tokenizer(models.BPE.from_file(
        os.path.join(DATA, "vocab.json"), os.path.join(DATA, "merges.txt")))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    assert tok.get_vocab_size() == 50257

    docs = []
    for t in TEXTS:
        ids = tok.encode(t).ids
        assert EOS not in ids, "EOS xuat hien trong doc!"
        docs.append(ids)

    # --- packing -----------------------------------------------------------
    seqs, docids = pack_sequences(docs, SEQ_LEN, EOS)
    flat = [t for s in seqs for t in s]

    # --- shard -------------------------------------------------------------
    n_shards = (len(flat) + SHARD_SIZE - 1) // SHARD_SIZE
    shards = [flat[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]
              for i in range(n_shards)]
    order42 = shard_order(n_shards, SEED_A)
    order43 = shard_order(n_shards, SEED_B)
    assert order42 != order43, "seed 42/43 cung thu tu — check nhanh!"

    def stream_samples(seed):
        """Sample tiep tuyen tu stream shard xep theo order(seed)."""
        stream = []
        for idx in shard_order(n_shards, seed):
            stream.extend(shards[idx])
        samples = []
        pos = 0
        while pos + SAMPLE_SEQ_LEN <= len(stream):
            samples.append(stream[pos:pos + SAMPLE_SEQ_LEN])
            pos += SAMPLE_SEQ_LEN
        return samples

    samples42 = stream_samples(SEED_A)
    samples43 = stream_samples(SEED_B)
    flat42 = [t for s in samples42 for t in s]
    flat43 = [t for s in samples43 for t in s]

    # --- ghi fixture (ASCII, format section) -------------------------------
    def ints(xs):
        return " ".join(str(x) for x in xs)

    lines = []
    lines.append("# G3 packing+shard fixture - sinh boi tools\\gen_pack_fixture.py")
    lines.append("# (reference impl, doi xung voi tv/data/dataset.tkv - cung hang so)")
    lines.append("#sec params")
    lines.append("seq_len=%d eos=%d shard_size=%d n_shards=%d seed_a=%d seed_b=%d "
                 "sample_seq_len=%d n_docs=%d"
                 % (SEQ_LEN, EOS, SHARD_SIZE, n_shards, SEED_A, SEED_B,
                    SAMPLE_SEQ_LEN, len(docs)))
    lines.append("#sec docs")
    for d in docs:
        lines.append(ints(d))
    lines.append("#sec pack_seqs")
    for s in seqs:
        lines.append(ints(s))
    lines.append("#sec pack_docids")
    for s in docids:
        lines.append(ints(s))
    lines.append("#sec mask_block_seq0")
    for row in block_causal_mask(docids[0]):
        lines.append(ints(row))
    lines.append("#sec mask_causal_4")
    for row in causal_mask(4):
        lines.append(ints(row))
    lines.append("#sec order_seed42")
    lines.append(ints(order42))
    lines.append("#sec order_seed43")
    lines.append(ints(order43))
    lines.append("#sec stream_samples_seed42")
    for s in samples42[:N_STREAM_SAMPLES_FIXTURE]:
        lines.append(ints(s))
    lines.append("#sec stream_checksum_seed42")
    lines.append(str(checksum(flat42)))
    lines.append("#sec stream_checksum_seed43")
    lines.append(str(checksum(flat43)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("-> %s (docs=%d seqs=%d n_shards=%d samples42=%d)"
          % (os.path.relpath(OUT, ROOT).replace("\\", "/"), len(docs),
             len(seqs), n_shards, len(samples42)))


if __name__ == "__main__":
    main()
