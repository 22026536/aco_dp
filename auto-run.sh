#!/bin/bash

set -e

instance_dir="normalized_instances/tsplib"
summary_dir="results/summary"

mkdir -p "$summary_dir"

for instance_path in "$instance_dir"/*.txt; do
    instance_file=$(basename "$instance_path")
    instance_name="${instance_file%.txt}"

    echo "=============================="
    echo "   Processing instance: $instance_name"
    echo "=============================="

    # Lấy số đầu tiên trong tên file để xác định time limit
    size=$(echo "$instance_name" | grep -oP '\d+' | head -1)

    if   [ "$size" -ge 1 ]   && [ "$size" -lt 100 ]; then
        time_limit=10
    elif [ "$size" -ge 100 ] && [ "$size" -lt 200 ]; then
        time_limit=60
    elif [ "$size" -ge 200 ] && [ "$size" -lt 400 ]; then
        time_limit=300
    elif [ "$size" -ge 400 ] && [ "$size" -lt 500 ]; then
        time_limit=1500
    else
        time_limit=1200  # mặc định nếu ngoài các khoảng trên
    fi

    echo "⏱️ Time limit cho $instance_name (size=$size): ${time_limit}s"

    # Tạo thư mục đích trước
    dest_logs="$summary_dir/$instance_name/logs"
    mkdir -p "$dest_logs"

    # --- RUN ACO ---
    echo "▶️ Chạy run-aco.sh cho $instance_name..."
    ./run-aco.sh "$instance_path" "$time_limit"

    # Copy logs sau khi chạy xong
    if [ -d "results/logs/$instance_name" ]; then
        cp -r "results/logs/$instance_name/." "$dest_logs/"
        echo "✅ Đã lưu logs vào $dest_logs"
    else
        echo "⚠️ Không tìm thấy logs cho $instance_name"
    fi

    echo "✅ Hoàn thành instance $instance_name"
    echo
done

echo "🎉 Tất cả instance đã được xử lý xong!"
