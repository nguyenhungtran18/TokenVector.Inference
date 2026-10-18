#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""So sanh 2 thu muc fixture o MOI DONG (byte-level sau khi norm CRLF).

9 chu so nghia trong (%.9g) dam bao round-trip cho binary32, nen dong text
giong nhau => bit giong nhau. Dung boi tools\\verify_parity.ps1 cap [1/3]:

    python tools/compare_fixtures.py <ref_dir> <new_dir>

Exit 0 = khac biet 0 file; exit 1 = co file thieu/khac/thua.
"""

import os
import sys


def load_lines(path):
    with open(path, "r", encoding="utf-8", newline=None) as f:
        return f.read().splitlines()


def main():
    if len(sys.argv) != 3:
        print("usage: compare_fixtures.py <ref_dir> <new_dir>")
        return 2
    ref_dir, new_dir = sys.argv[1], sys.argv[2]
    ref = sorted(os.listdir(ref_dir)) if os.path.isdir(ref_dir) else []
    new = sorted(os.listdir(new_dir)) if os.path.isdir(new_dir) else []

    same, diff, missing, extra = [], [], [], []
    for name in ref:
        if not name.endswith(".txt"):
            continue
        if name not in new:
            missing.append(name)
            continue
        a = load_lines(os.path.join(ref_dir, name))
        b = load_lines(os.path.join(new_dir, name))
        if a == b:
            same.append(name)
        else:
            diff.append((name, a, b))

    for name in new:
        if name.endswith(".txt") and name not in ref:
            extra.append(name)

    print("== compare_fixtures: ref=%s vs new=%s" % (ref_dir, new_dir))
    print("  khop: %d file" % len(same))
    for name, a, b in diff:
        idx = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), 0)
        ra = a[idx][:70] if idx < len(a) else "<khong co>"
        rb = b[idx][:70] if idx < len(b) else "<khong co>"
        print("  [KHAC] %s (dong %d)" % (name, idx + 1))
        print("        ref: %s" % ra)
        print("        new: %s" % rb)
    for name in missing:
        print("  [THIEU] %s (khong duoc sinh lai)" % name)
    for name in extra:
        print("  [THUA] %s (chua check-in)" % name)

    bad = len(diff) + len(missing) + len(extra)
    print("  ket qua: %s (diff=%d missing=%d extra=%d)"
          % ("KHOP" if bad == 0 else "LECH", len(diff), len(missing), len(extra)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
