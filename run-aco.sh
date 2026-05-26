#!/bin/bash
# =============================================================================
# run-aco.sh — Chạy ACO một lần trên một instance
#
# Cách dùng:
#   ./run-aco.sh <instance_path> [max_time] [max_iter]
#
# Ví dụ:
#   ./run-aco.sh data/inst_100_5.txt 300 5000
# =============================================================================

set -euo pipefail

# ─── Tham số đầu vào ─────────────────────────────────────────────────────────
INSTANCE_PATH="${1:-}"
MAX_TIME="${2:-1500}"
MAX_ITER="${3:-10000}"

if [ -z "$INSTANCE_PATH" ]; then
    echo "❌  Thiếu đường dẫn instance!"
    echo "    Cách dùng: $0 <instance_path> [max_time] [max_iter]"
    exit 1
fi

if [ ! -f "$INSTANCE_PATH" ]; then
    echo "❌  Không tìm thấy file: $INSTANCE_PATH"
    exit 1
fi

if [ ! -f "./MCGP" ]; then
    echo "❌  Không tìm thấy binary ./MCGP (cần compile trước)"
    exit 1
fi

# ─── Tên instance ─────────────────────────────────────────────────────────────
INSTANCE_NAME=$(basename "$INSTANCE_PATH" | sed 's/\.[^.]*$//')

# ─── Chuẩn bị thư mục log ────────────────────────────────────────────────────
if [ -d "results/logs/${INSTANCE_NAME}" ]; then
    chmod -R u+rwx "results/logs/${INSTANCE_NAME}" 2>/dev/null || true
    rm -rf "results/logs/${INSTANCE_NAME}"
fi
mkdir -p "results/logs/${INSTANCE_NAME}/evolution"
mkdir -p "results/logs/${INSTANCE_NAME}/solutions"
mkdir -p "results/logs/${INSTANCE_NAME}/objectives"

echo "▶  Chạy ACO trên instance : $INSTANCE_NAME"
echo "   Time limit             : ${MAX_TIME}s"
echo "   Max iterations         : ${MAX_ITER}"
echo ""

./MCGP \
    --instance          "$INSTANCE_PATH" \
    --termination_value "$MAX_TIME"      \
    --iter_value        "$MAX_ITER"      \
    --logs              10
