#!/bin/bash
# Gate 135/135 — 9 suite nho (102) + gpt2_full (33), compiler tkvc-tv.exe 2026-10-26
# Cach dung: bash tools/gate_f32.sh
set -o pipefail
TKVC=/d/TokenVector/Release/dist/tkvc-tv.exe
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 1
mkdir -p out
fails_total=0
declare -a SUMMARY

run_suite() {
  local name="$1" src="$2" expect="$3"
  echo "== $name =="
  if ! "$TKVC" "tv/tests/$src" "out/g_$name.exe" main > "out/gate_$name.build.log" 2>&1; then
    SUMMARY+=("$name: BUILD FAIL")
    fails_total=$((fails_total + 1))
    tail -3 "out/gate_$name.build.log"
    return
  fi
  local out
  out=$("out/g_$name.exe" 2>&1)
  echo "$out" | tail -2
  local f
  f=$(echo "$out" | grep -oE 'fails=[0-9]+' | grep -oE '[0-9]+')
  if [ -z "$f" ]; then
    # suite khong in fails= (vd gap: "PASS")
    if echo "$out" | grep -q "PASS"; then f=0; else f=1; fi
  fi
  fails_total=$((fails_total + f))
  SUMMARY+=("$name: fails=$f (expect $expect)")
}

run_suite simd_matmul_tests simd_matmul_tests.tkv 0
run_suite gpt2_tests        gpt2_tests.tkv        0
run_suite gpt2_scope_tests  gpt2_scope_tests.tkv  0
run_suite gpt2_train_tests  gpt2_train_tests.tkv  0
run_suite grad_tests        grad_tests.tkv        0
run_suite parity_tests      parity_tests.tkv      0
run_suite pack_tests        pack_tests.tkv        0
run_suite inplace_tests     inplace_tests.tkv     0
run_suite gap_tests         gap_tests.tkv         0
run_suite gpt2_full_tests   gpt2_full_tests.tkv   0

echo
echo "================ GATE SUMMARY (tkvc 2026-10-26) ================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "  TOTAL fails=$fails_total (target 0 = 135/135)"
echo "==============================================================="
exit $fails_total