#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture TOKEN ID cho G3 (tokenizer parity GPT-2 BPE).

Dung chinh vocab.json/merges.txt da tai ve tv\\tests\\data\\ (ban goc tu
HuggingFace openai-community/gpt2) de mo GPT2TokenizerFast OFFLINE — khong
can mang luc chay.

Moi ca: 2 dong trong tv\\tests\\fixtures\\tok_cases.txt
    dong le:   text da escape (\\n, \\t, \\r, \\\\)
    dong chan: cac token id cach nhau boi dau cach
Dong dau file la comment '# ...'.

Chay:  python tools\\gen_tokenizer_fixture.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "tv", "tests", "data")
OUT = os.path.join(ROOT, "tv", "tests", "fixtures", "tok_cases.txt")

# Cac doan van bao gom: ASCII, whitespace/contract, dau cau, so, tieng Viet,
# unicode/emoji, newline/tab, canh le. Moi doan la 1 ca.
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
    " ",
    "",
]


def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def main():
    # ORACLE: tokenizers (Rust — chinh backend cua HF GPT2TokenizerFast) va
    # tiktoken (gpt2 cua OpenAI) phai KHOP NHAU tung id truoc khi ghi fixture.
    # (transformers 5.x khong nac duoc GPT2Tokenizer tu vocab_file/merges_file
    # — len=1 — nen dung truc tiep 2 lib reference.)
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    except ImportError:
        print("CANH BAO: khong tim thay tokenizers.")
        sys.exit(2)
    try:
        import tiktoken
    except ImportError:
        print("CANH BAO: khong tim thay tiktoken (can de cross-check oracle).")
        sys.exit(2)

    tok = Tokenizer(models.BPE.from_file(
        os.path.join(DATA, "vocab.json"), os.path.join(DATA, "merges.txt")))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    n_vocab = tok.get_vocab_size()
    assert n_vocab == 50257, "vocab size %d != 50257" % n_vocab
    tk = tiktoken.get_encoding("gpt2")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = ["# GPT-2 BPE token ids tu HF GPT2TokenizerFast (vocab/merges chinh "
             "goc openai-community/gpt2); dong le = text escape, dong chan = ids"]
    for i, text in enumerate(TEXTS):
        ids = tok.encode(text).ids
        ids_tk = tk.encode(text)
        assert ids == ids_tk, (
            "oracle lech nhau o case %d: tokenizers=%s tiktoken=%s"
            % (i, ids[:12], ids_tk[:12]))
        lines.append(esc(text))
        lines.append(" ".join(str(x) for x in ids))
        print("  case %d: len(text)=%d n_ids=%d (tokenizers==tiktoken)"
              % (i, len(text), len(ids)))

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("-> %s (%d ca)" % (os.path.relpath(OUT, ROOT), len(TEXTS)))


if __name__ == "__main__":
    main()
