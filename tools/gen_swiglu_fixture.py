#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture cho op SwiGLU (G1 — dong bo ROADMAP "swiglu").

Input [n, 2d]: nua dau GATE (qua silu), nua sau VALUE.
    y = silu(gate) * value
Gradient lay bang torch.autograd voi gy cho truoc.

Chay:  python tools\\gen_swiglu_fixture.py
"""

import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tv", "tests", "fixtures")

N = 4
D = 8


def write_case(name, header, values):
    path = os.path.join(OUT, "g_sw_" + name + ".txt")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# n=%d  %s\n" % (len(values), header))
        f.write(" ".join("%.9g" % float(v) for v in values) + "\n")
    print("  -> g_sw_%-6s n=%d" % (name, len(values)))


def main():
    torch.manual_seed(11)
    x = torch.randn(N, 2 * D, dtype=torch.float32)
    gate = x[:, :D].clone().requires_grad_(True)
    val = x[:, D:].clone().requires_grad_(True)
    y = F.silu(gate) * val
    gy = torch.randn(N, D, dtype=torch.float32)
    y.backward(gy)

    write_case("x", "swiglu input [n,2d]", x.reshape(-1).tolist())
    write_case("y", "silu(gate)*value", y.detach().reshape(-1).tolist())
    write_case("gy", "grad output", gy.reshape(-1).tolist())
    gx = torch.cat([gate.grad, val.grad], dim=1)
    write_case("gx", "grad input [n,2d]", gx.reshape(-1).tolist())
    print("done. n=%d d=%d" % (N, D))


if __name__ == "__main__":
    main()
