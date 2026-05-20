#!/bin/bash
# =============================================================================
# auto-run-benchmark.sh — Tự động chạy run-benchmark.sh cho toàn bộ instance
# =============================================================================

set -euo pipefail

# ─── Tham số đầu vào ─────────────────────────────────────────────────────────
INSTANCE_DIR="${1:-}"
NUM_RUNS="${2:-10}"
TIME_LIMIT_OVERRIDE="${3:-}"   # nếu rỗng → tự động theo n

if [ -z "$INSTANCE_DIR" ]; then
    echo "❌  Thiếu thư mục instance!"
    echo "    Cách dùng: $0 <instance_dir> [num_runs] [time_limit]"
    exit 1
fi

if [ ! -d "$INSTANCE_DIR" ]; then
    echo "❌  Không tìm thấy thư mục: $INSTANCE_DIR"
    exit 1
fi

if [ ! -f "./run-benchmark.sh" ]; then
    echo "❌  Không tìm thấy run-benchmark.sh trong thư mục hiện tại"
    exit 1
fi

if [ ! -f "./MCGP" ]; then
    echo "❌  Không tìm thấy binary ./MCGP (cần compile trước)"
    exit 1
fi

chmod +x ./run-benchmark.sh

# ─── Màu sắc ─────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'; DIM='\033[2m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''; DIM=''
fi

# ─── Hàm: tự động chọn time limit theo n ────────────────────────────────────
auto_time_limit() {
    local instance_path="$1"
    local instance_file
    instance_file=$(basename "$instance_path")
    local n
    n=$(echo "$instance_file" | grep -oP '\d+' | head -1)

    if   [ "$n" -ge 1 ]   && [ "$n" -lt 100 ]; then echo 10
    elif [ "$n" -ge 100 ] && [ "$n" -lt 200 ]; then echo 60
    elif [ "$n" -ge 200 ] && [ "$n" -lt 400 ]; then echo 300
    elif [ "$n" -ge 400 ] && [ "$n" -lt 500 ]; then echo 1500
    else                                             echo 1200
    fi
}

# ─── Thu thập danh sách instance ─────────────────────────────────────────────
mapfile -t INSTANCES < <(find "$INSTANCE_DIR" -maxdepth 1 -type f | sort -V)
TOTAL=${#INSTANCES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "❌  Không tìm thấy file nào trong: $INSTANCE_DIR"
    exit 1
fi

# ─── File log tổng hợp ───────────────────────────────────────────────────────
mkdir -p "results/benchmark"
GLOBAL_CSV="results/benchmark/_summary_all.csv"

if [ ! -f "$GLOBAL_CSV" ]; then
    echo "instance,n,feasible_runs,best_cost,avg_cost,std_cost,avg_time_to_best_s,time_limit_s" \
         > "$GLOBAL_CSV"
fi

# ─── Hàm: append stats vào global CSV ───────────────────────────────────────
append_to_global() {
    local instance_name="$1"
    local instance_path="$2"
    local stats_file="$3"
    local time_limit="$4"

    # Xóa dòng cũ để overwrite
    sed -i "/^${instance_name},/d" "$GLOBAL_CSV"

    local n
    n=$(head -1 "$instance_path" | awk '{print $1}')

    local feasible_runs
    feasible_runs=$(grep "Feasible runs:" "$stats_file" | awk '{print $3}')

    local best_cost
    best_cost=$(grep "Best cost:" "$stats_file" | awk '{print $3}')

    local avg_cost
    avg_cost=$(grep "Avg cost:" "$stats_file" | awk '{print $3}')

    local std_cost
    std_cost=$(grep "Std dev cost:" "$stats_file" | awk '{print $4}')

    local avg_time
    avg_time=$(grep "Avg time-to-best:" "$stats_file" | awk '{print $3}')

    echo "${instance_name},${n},${feasible_runs},${best_cost},${avg_cost},${std_cost},${avg_time},${time_limit}" \
        >> "$GLOBAL_CSV"
}

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Auto Benchmark — MCGP                                   ║${RESET}"
echo -e "${BOLD}║  Thư mục : ${INSTANCE_DIR}${RESET}"
echo -e "${BOLD}║  Số lần  : ${NUM_RUNS}    Time limit: ${TIME_LIMIT_OVERRIDE:-tự động theo n}s${RESET}"
echo -e "${BOLD}║  Tổng    : ${TOTAL} instance${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─── Biến đếm ────────────────────────────────────────────────────────────────
DONE=0
SKIPPED=0
FAILED=0
WALL_START=$(date +%s)

# =============================================================================
# VÒNG LẶP CHÍNH
# =============================================================================
for INSTANCE_PATH in "${INSTANCES[@]}"; do

    INSTANCE_NAME=$(basename "$INSTANCE_PATH")
    INSTANCE_NAME="${INSTANCE_NAME%.*}"
    RESULT_DIR="results/benchmark/${INSTANCE_NAME}"
    STATS_FILE="${RESULT_DIR}/stats.txt"

    # ── Time limit ───────────────────────────────────────────────────────────
    if [ -n "$TIME_LIMIT_OVERRIDE" ]; then
        TIME_LIMIT="$TIME_LIMIT_OVERRIDE"
    else
        TIME_LIMIT=$(auto_time_limit "$INSTANCE_PATH")
    fi

    IDX=$((DONE + SKIPPED + FAILED + 1))
    echo -e "${BOLD}[${IDX}/${TOTAL}]${RESET} ${CYAN}${INSTANCE_NAME}${RESET}  ${DIM}(limit=${TIME_LIMIT}s)${RESET}"

    # ── Skip nếu đã có kết quả ──────────────────────────────────────────────
    if [ -f "$STATS_FILE" ]; then
        echo -e "  ${YELLOW}↻ Đã có stats.txt — chạy lại (overwrite)${RESET}"
        rm -rf "$RESULT_DIR"
    fi

    # ── Run benchmark ───────────────────────────────────────────────────────
    set +e
    ./run-benchmark.sh "$INSTANCE_PATH" "$NUM_RUNS" "$TIME_LIMIT"
    RUN_EXIT=$?
    set -e

    if [ $RUN_EXIT -ne 0 ]; then
        echo -e "  ${RED}✗  run-benchmark.sh thất bại (exit=${RUN_EXIT})${RESET}"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    DONE=$((DONE + 1))

    # ── Append kết quả ──────────────────────────────────────────────────────
    if [ -f "$STATS_FILE" ]; then
        append_to_global "$INSTANCE_NAME" "$INSTANCE_PATH" "$STATS_FILE" "$TIME_LIMIT"
    fi

    # ── ETA ─────────────────────────────────────────────────────────────────
    WALL_NOW=$(date +%s)
    ELAPSED=$((WALL_NOW - WALL_START))
    COMPLETED=$((DONE + FAILED + SKIPPED))

    if [ "$COMPLETED" -gt 0 ]; then
        AVG_SEC_PER=$(( ELAPSED / COMPLETED ))
        REMAINING=$(( (TOTAL - IDX) * AVG_SEC_PER ))
        echo -e "  ${DIM}⏱  Đã chạy ${ELAPSED}s  |  Ước tính còn ~${REMAINING}s${RESET}"
    fi
    echo ""

done

# ─── Tổng kết ───────────────────────────────────────────────────────────────
WALL_END=$(date +%s)
TOTAL_TIME=$((WALL_END - WALL_START))

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Hoàn tất!${RESET}"
printf "  %-22s %d\n"  "Đã chạy mới:"   "$DONE"
printf "  %-22s %d\n"  "Bỏ qua (cache):" "$SKIPPED"
printf "  %-22s %d\n"  "Thất bại:"       "$FAILED"
printf "  %-22s %ds\n" "Tổng thời gian:" "$TOTAL_TIME"
echo ""
echo -e "  ${GREEN}✔  Kết quả tổng hợp: ${GLOBAL_CSV}${RESET}"
echo ""
