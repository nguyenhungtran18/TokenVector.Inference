#!/bin/bash
# Gate 135/135 — 9 suite nho (102) + gpt2_full (33), compiler tkvc-tv.exe
# Cach dung: bash tools/gate_f32.sh
#
# RUNTIME (2026-10-30): ban dich co IL vector P6 (`System.Numerics.Vector`1`
# + `Vector.Widen/Narrow`). .NET Framework khong co `Widen` (nguoi goc
# `System.Numerics.Vectors` trong GAC la reference assembly rong) ->
# `MissingMethodException`/`TypeLoadException` NGAY GIUA test. Chun `run_bin`
# chay tren .NET Framework truoc, neu thay loi do thi chay lai tren .NET 10
# (runtime co `Vector.Widen`/`Narrow` hardware) - xem HANDOFF §0.56.
#
# #9 (2026-10-30): ban dich con goi intrinsic AVX truc tiep
# (`System.Runtime.Intrinsics`) cho Widen/Narrow - .NET Framework cung khong
# co assembly do -> mo rong dieu kien nhan dien marker thu 2. Chi cac file
# THUC SU vector hoa f64 moi tham chieu (emit lazy), cac suite khac van chay
# tren .NET Framework nhu cu.
#
# CHAT NGHIEM (khong an loi): mot suite chi TINH pass khi CUNG LUC:
#   (1) exit code = 0,  (2) khong co "Unhandled Exception",
#   (3) co `fails=0` (hoac PASS doi voi suite khong in fails=).
# Truoc day chi doc `fails=`/`PASS` tu output nen mot suite rot giua
# van tinh pass = 0 — da sua.
#
# KHONG chinh sua file nay KHI no dang chay (bash doc script theo tung dot,
# sua se lam sai offset dong).
set -o pipefail
TKVC=/d/TokenVector/Release/dist/tkvc-tv.exe
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 1
mkdir -p out
fails_total=0
declare -a SUMMARY

# runtimeconfig cho `dotnet <dll>` - tao 1 lan/moi ten, lay phien ban .NET da cai.
make_n10_cfg() {
  local cfg="$1"
  local rt
  rt=$(dotnet --list-runtimes 2>/dev/null | grep 'Microsoft.NETCore.App ' | tail -1 \
       | sed -E 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/')
  if [ -z "$rt" ]; then
    return 1
  fi
  printf '{\n  "runtimeOptions": {\n    "tfm": "net10.0",\n    "framework": {\n      "name": "Microsoft.NETCore.App",\n      "version": "%s"\n    }\n  }\n}\n' \
    "$rt" > "out/${cfg}.runtimeconfig.json"
  return 0
}

# Chay 1 da dich. Ket qua ghi ra out/gate_<name>.out, exit code ra
# out/gate_<name>.ec. Neu .NET Framework khong chay duoc Vector`1/Widen thi
# chay lai tren .NET 10 (luc do ec/out la cua phien .NET 10).
run_bin() {
  local name="$1" out ec
  out=$(  "out/g_$name.exe" 2>&1); ec=$?
  if printf '%s' "$out" | grep -qE "System\.Numerics|System\.Runtime\.Intrinsics" \
     && printf '%s' "$out" | grep -qE "TypeLoadException|MissingMethodException|FileNotFoundException"; then
    if make_n10_cfg "g_${name}_n10" 2>/dev/null; then
      cp -f "out/g_$name.exe" "out/g_${name}_n10.dll"
      out=$(dotnet "out/g_${name}_n10.dll" 2>&1); ec=$?
    fi
  fi
  printf '%s' "$out" > "out/gate_$name.out"
  printf '%s' "$ec" > "out/gate_$name.ec"
}

run_suite() {
  local name="$1" src="$2" expect="$3"
  echo "== $name =="
  if ! "$TKVC" "tv/tests/$src" "out/g_$name.exe" main > "out/gate_$name.build.log" 2>&1; then
    SUMMARY+=("$name: BUILD FAIL")
    fails_total=$((fails_total + 1))
    tail -3 "out/gate_$name.build.log"
    return
  fi
  run_bin "$name"
  local out ec f="" why=""
  out=$(cat "out/gate_$name.out")
  ec=$(cat "out/gate_$name.ec")
  echo "$out" | tail -2

  # (1) exit code  (2) khong rot
  if [ "$ec" != "0" ]; then why="exit=$ec"; fi
  if printf '%s' "$out" | grep -q "Unhandled Exception"; then
    why="${why:+$why+}crash"
  fi
  if [ -n "$why" ]; then
    f=1
    SUMMARY+=("$name: FAIL ($why)")
    fails_total=$((fails_total + 1))
    return
  fi
  # (3) ket qua ro rang
  f=$(printf '%s' "$out" | grep -oE 'fails=[0-9]+' | grep -oE '[0-9]+' | head -1)
  if [ -z "$f" ]; then
    if printf '%s' "$out" | grep -qE '\[PASS\]|PASS'; then f=0; else f=1; fi
  fi
  if [ "$f" != "0" ]; then
    SUMMARY+=("$name: FAIL fails=$f")
    fails_total=$((fails_total + 1))
    return
  fi
  SUMMARY+=("$name: PASS")
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
echo "================ GATE SUMMARY (tkvc 2026-10-30) ================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "  TOTAL fails=$fails_total (target 0 = 135/135)"
echo "==============================================================="
exit $fails_total
