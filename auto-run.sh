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

    # # --- RUN PRED ---
    # echo "▶️ Chạy run.sh (pred) cho $instance_name..."
    # ./run.sh "$instance_path" &
    # pid_pred=$!

    # # --- RUN DP ---
    # echo "▶️ Chạy run-dp.sh cho $instance_name..."
    # ./run-dp.sh "$instance_path" &
    # pid_dp=$!

    # # --- RUN HDP ---
    # echo "▶️ Chạy run-hdp.sh cho $instance_name..."
    # ./run-hdp.sh "$instance_path" &
    # pid_hdp=$!

    # --- RUN ACO ---
    echo "▶️ Chạy run-aco.sh (pred) cho $instance_name..."
    ./run-aco.sh "$instance_path" &
    pid_aco=$!

    # Chờ cả 4 tiến trình xong
    wait $pid_aco

    # Sau khi tất cả đã xong, copy kết quả
    # dest_pred="$summary_dir/$instance_name/pred_logs"
    # mkdir -p "$dest_pred"
    # cp -r results/logs/pred_logs/* "$dest_pred"/

    # dest_dp="$summary_dir/$instance_name/dp_logs"
    # mkdir -p "$dest_dp"
    # cp -r results/logs/dp_logs/* "$dest_dp"/

    # dest_hdp="$summary_dir/$instance_name/hdp_logs"
    # mkdir -p "$dest_hdp"
    # cp -r results/logs/hdp_logs/* "$dest_hdp"/

    dest_aco="$summary_dir/$instance_name/aco_logs"
    mkdir -p "$dest_aco"
    cp -r results/logs/aco_logs/* "$dest_aco"/

    echo "✅ Hoàn thành instance $instance_name"
    echo
done

echo "🎉 Tất cả instance đã được xử lý xong!"
