#!/bin/bash
# =============================================================================
# auto-run.sh — Chạy ACO trên toàn bộ instance trong một thư mục
#
# Cách dùng:
#   ./auto-run.sh <instance_dir> [max_time_override] [max_iter]
#
# Ví dụ:
#   ./auto-run.sh instances/pollster              # time limit tự động theo kích thước
#   ./auto-run.sh instances/pollster 60           # ép tất cả dùng 60s
#   ./auto-run.sh instances/pollster 60 5000      # ép 60s, tối đa 5000 iter
# =============================================================================

set -e
shopt -s nullglob

# ─── Tham số đầu vào ─────────────────────────────────────────────────────────
instance_dir="${1:-}"
time_override="${2:-}"
max_iter="${3:-10000}"

if [ -z "$instance_dir" ]; then
    echo "❌  Thiếu thư mục instance!"
    echo "    Cách dùng: $0 <instance_dir> [max_time_override] [max_iter]"
    exit 1
fi

if [ ! -d "$instance_dir" ]; then
    echo "❌  Không tìm thấy thư mục: $instance_dir"
    exit 1
fi

summary_dir="results/summary"
mkdir -p "$summary_dir"

# ─── Kiểm tra có file .txt không ─────────────────────────────────────────────
files=("$instance_dir"/*.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "⚠️  Không tìm thấy file .txt nào trong: $instance_dir"
    exit 0
fi

# ─── Duyệt từng instance ─────────────────────────────────────────────────────
for instance_path in "${files[@]}"; do
    instance_file=$(basename "$instance_path")
    instance_name="${instance_file%.txt}"

    echo "=============================="
    echo "   Processing instance: $instance_name"
    echo "=============================="

    # Lấy số đầu tiên trong tên file để xác định time limit tự động
    size=$(echo "$instance_name" | grep -oP '\d+' | head -1)

    if [ -n "$time_override" ]; then
        time_limit="$time_override"
    elif [ -z "$size" ]; then
        time_limit=1200
    elif [ "$size" -ge 1 ]   && [ "$size" -lt 100 ]; then
        time_limit=10
    elif [ "$size" -ge 100 ] && [ "$size" -lt 200 ]; then
        time_limit=60
    elif [ "$size" -ge 200 ] && [ "$size" -lt 400 ]; then
        time_limit=300
    elif [ "$size" -ge 400 ] && [ "$size" -lt 500 ]; then
        time_limit=1500
    else
        time_limit=1200
    fi

    echo "⏱️  Time limit cho $instance_name (size=${size:-?}): ${time_limit}s  |  Max iter: ${max_iter}"

    dest_logs="$summary_dir/$instance_name/logs"
    mkdir -p "$dest_logs"

    echo "▶️  Chạy run-aco.sh cho $instance_name..."
    ./run-aco.sh "$instance_path" "$time_limit" "$max_iter"

    if [ -d "results/logs/$instance_name" ]; then
        cp -r "results/logs/$instance_name/." "$dest_logs/"
        echo "✅ Đã lưu logs vào $dest_logs"
    else
        echo "⚠️  Không tìm thấy logs cho $instance_name"
    fi

    echo "✅ Hoàn thành instance $instance_name"
    echo
done

echo "🎉 Tất cả instance đã được xử lý xong!"
