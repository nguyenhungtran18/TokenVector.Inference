#!/bin/bash
# Chay tools/gate_f32.sh trong WSL.
#
# Vi sao can script nay (2026-11-10):
#  1. gate_f32.sh co CRLF -> bash trong WSL chet ngay dong `set -o pipefail`.
#     Script nay tu chuyen sang LF roi moi chay.
#  2. WSL khong co `dotnet`. Gate can .NET 10 de fallback khi ban dich cham
#     System.Runtime.Intrinsics (may nay LUON cham o gpt2_layernorm) -> khong co
#     dotnet thi 8/10 suite bao FAIL gia. Script nay cai shim tro sang dotnet.exe
#     cua Windows.
#
# Dung: wsl -e bash /mnt/d/TokenVector.Inference/tools/_rungate.sh
set -u

DNET="/mnt/c/Users/Admin/.dotnet/dotnet.exe"
REPO="/mnt/d/TokenVector.Inference"

if [ ! -x "$DNET" ]; then
  echo "FATAL: khong thay $DNET"
  exit 1
fi

# shim dotnet cho WSL (root nen ghi duoc /usr/local/bin)
if ! command -v dotnet >/dev/null 2>&1; then
  printf '#!/bin/sh\nexec "%s" "$@"\n' "$DNET" > /usr/local/bin/dotnet 2>/dev/null
  chmod +x /usr/local/bin/dotnet 2>/dev/null
fi
if ! command -v dotnet >/dev/null 2>&1; then
  echo "FATAL: khong cai duoc shim dotnet (can quyen ghi /usr/local/bin)"
  exit 1
fi
echo "dotnet -> $(command -v dotnet)"
dotnet --list-runtimes 2>/dev/null | grep 'Microsoft.NETCore.App ' | tail -1

cd "$REPO" || exit 1

# gate_f32.sh trong repo la CRLF -> tao ban LF tam
TMP="tools/_gate_lf.sh"
sed 's/\r$//' tools/gate_f32.sh > "$TMP" || exit 1
trap 'rm -f "$TMP"' EXIT

bash "$TMP" 2>&1 | tail -25
