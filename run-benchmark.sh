#!/bin/bash
# =============================================================================
# run-benchmark.sh — Chạy MCGP nhiều lần, ghi kết quả và tính thống kê
#
# Cách dùng:
#   ./run-benchmark.sh <instance_path> [num_runs] [max_time] [max_iter]
#
# Ví dụ:
#   ./run-benchmark.sh data/inst_100_5.txt 10 60 5000
#
# Output (trong results/benchmark/<instance_name>/):
#   run_N.log   — stdout mỗi lần chạy  (Final cost, feasibility)
#   run_N.err   — stderr mỗi lần chạy  (ITER logs, time-to-best)
#   summary.csv — Best, feasible, time-to-best từng lần
#   stats.txt   — Avg/Best/Std tổng hợp
# =============================================================================

set -euo pipefail

# ─── Tham số đầu vào ─────────────────────────────────────────────────────────
INSTANCE_PATH="${1:-}"
NUM_RUNS="${2:-10}"
MAX_TIME="${3:-1500}"
MAX_ITER="${4:-10000}"

if [ -z "$INSTANCE_PATH" ]; then
    echo "❌  Thiếu đường dẫn instance!"
    echo "    Cách dùng: $0 <instance_path> [num_runs] [max_time] [max_iter]"
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

# ─── Thư mục lưu kết quả benchmark ───────────────────────────────────────────
RESULT_DIR="results/benchmark/${INSTANCE_NAME}"
mkdir -p "$RESULT_DIR"

# ─── Màu terminal ─────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
    BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; CYAN=''; BOLD=''; RESET=''
fi

# ─── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  MCGP Benchmark — ${INSTANCE_NAME}${RESET}"
echo -e "${BOLD}  Runs: ${NUM_RUNS}   Time limit: ${MAX_TIME}s   Max iter: ${MAX_ITER}${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo ""

# ─── File tổng hợp CSV ────────────────────────────────────────────────────────
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "run,seed,feasible,best_cost,time_to_best_s" > "$SUMMARY_CSV"

declare -a ALL_COSTS=()
declare -a ALL_TIMES=()

# =============================================================================
# VÒNG LẶP CHÍNH
# =============================================================================
for (( RUN=1; RUN<=NUM_RUNS; RUN++ )); do

    SEED=$((1000000000 + RUN))

    LOG_FILE="${RESULT_DIR}/run_${RUN}.log"
    ERR_FILE="${RESULT_DIR}/run_${RUN}.err"

    echo -e "${CYAN}▶  Run ${RUN}/${NUM_RUNS}${RESET}  (seed=${SEED})"

    # Dọn thư mục log của lần chạy trước
    if [ -d "results/logs/${INSTANCE_NAME}" ]; then
        chmod -R u+rwx "results/logs/${INSTANCE_NAME}" 2>/dev/null || true
        rm -rf "results/logs/${INSTANCE_NAME}"
    fi
    mkdir -p "results/logs/${INSTANCE_NAME}/evolution"
    mkdir -p "results/logs/${INSTANCE_NAME}/solutions"
    mkdir -p "results/logs/${INSTANCE_NAME}/objectives"

    # ── Chạy MCGP ─────────────────────────────────────────────────────────────
    set +e
    ./MCGP \
        --instance          "$INSTANCE_PATH" \
        --seed              "$SEED"          \
        --termination_value "$MAX_TIME"      \
        --iter_value        "$MAX_ITER"      \
        --logs              10               \
        > "$LOG_FILE"                        \
        2> "$ERR_FILE"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "  ${RED}✗  Lần chạy thất bại (exit code ${EXIT_CODE})${RESET}"
        echo -e "  ${RED}    Xem chi tiết: cat ${ERR_FILE}${RESET}"
        head -5 "$ERR_FILE" 2>/dev/null | sed 's/^/      /' || true
        echo "${RUN},${SEED},ERROR,NA,NA" >> "$SUMMARY_CSV"
        ALL_COSTS+=("NA")
        ALL_TIMES+=("NA")
        continue
    fi

    # ── Parse kết quả ─────────────────────────────────────────────────────────
    FINAL_COST=$(grep -oP '(?<=Final cost = )[\d,]+\.?\d*' "$LOG_FILE" \
                 | tail -1 | tr -d ',')

    if grep -q "Final solution is valid" "$LOG_FILE"; then
        FEASIBLE="YES"
    else
        FEASIBLE="NO"
    fi

    TIME_TO_BEST=$(grep -oP 'time \K[\d.]+(?=s\))' "$ERR_FILE" | tail -1)
    [ -z "$TIME_TO_BEST" ] && TIME_TO_BEST="NA"

    if [ -z "$FINAL_COST" ]; then
        echo -e "  ${RED}✗  Không parse được cost từ output${RESET}"
        echo "${RUN},${SEED},${FEASIBLE},NA,NA" >> "$SUMMARY_CSV"
        ALL_COSTS+=("NA")
        ALL_TIMES+=("NA")
        continue
    fi

    # ── In kết quả lần này ────────────────────────────────────────────────────
    FEAS_COLOR="$RED"
    [ "$FEASIBLE" = "YES" ] && FEAS_COLOR="$GREEN"

    echo -e "  cost           = ${BOLD}${FINAL_COST}${RESET}"
    echo -e "  feasible       = ${FEAS_COLOR}${FEASIBLE}${RESET}"
    echo -e "  time-to-best   = ${TIME_TO_BEST}s"
    echo ""

    echo "${RUN},${SEED},${FEASIBLE},${FINAL_COST},${TIME_TO_BEST}" >> "$SUMMARY_CSV"
    ALL_COSTS+=("$FINAL_COST")
    ALL_TIMES+=("$TIME_TO_BEST")

done

# =============================================================================
# THỐNG KÊ TỔNG HỢP
# =============================================================================
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Kết quả tổng hợp${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"

STATS=$(awk -F',' '
NR == 1 { next }
$4 == "NA" || $5 == "NA" { next }
{
    cost = $4 + 0; time = $5 + 0; n++
    sum_cost += cost; sum_time += time
    if (n == 1 || cost < min_cost) { min_cost = cost; min_time = time }
    if (n == 1 || cost > max_cost)   max_cost = cost
    costs[n] = cost; times[n] = time
}
END {
    if (n == 0) { print "NO_VALID_RUNS"; exit }
    avg_cost = sum_cost / n; avg_time = sum_time / n
    for (i = 1; i <= n; i++) var_cost += (costs[i] - avg_cost)^2
    std_cost = (n > 1) ? sqrt(var_cost / (n-1)) : 0
    printf "valid_runs=%d\n",   n
    printf "best_cost=%.6f\n",  min_cost
    printf "worst_cost=%.6f\n", max_cost
    printf "avg_cost=%.6f\n",   avg_cost
    printf "std_cost=%.6f\n",   std_cost
    printf "avg_time=%.4f\n",   avg_time
    printf "best_time=%.4f\n",  min_time
}
' "$SUMMARY_CSV")

if echo "$STATS" | grep -q "NO_VALID_RUNS"; then
    echo -e "${RED}❌  Không có lần chạy nào thành công!${RESET}"
    exit 1
fi

N_FEASIBLE=$(awk   -F',' 'NR>1 && $3=="YES"   {n++} END {print n+0}' "$SUMMARY_CSV")
N_INFEASIBLE=$(awk -F',' 'NR>1 && $3=="NO"    {n++} END {print n+0}' "$SUMMARY_CSV")
N_ERROR=$(awk      -F',' 'NR>1 && $3=="ERROR" {n++} END {print n+0}' "$SUMMARY_CSV")

VALID_RUNS=$(echo "$STATS" | grep valid_runs  | cut -d= -f2)
BEST_COST=$(echo  "$STATS" | grep ^best_cost  | cut -d= -f2)
WORST_COST=$(echo "$STATS" | grep ^worst_cost | cut -d= -f2)
AVG_COST=$(echo   "$STATS" | grep ^avg_cost   | cut -d= -f2)
STD_COST=$(echo   "$STATS" | grep ^std_cost   | cut -d= -f2)
AVG_TIME=$(echo   "$STATS" | grep ^avg_time   | cut -d= -f2)
BEST_TIME=$(echo  "$STATS" | grep ^best_time  | cut -d= -f2)

printf "  %-24s %s / %s\n"  "Valid runs:"       "$VALID_RUNS"   "$NUM_RUNS"
printf "  %-24s %s feasible,  %s infeasible,  %s error\n" \
                             "Feasibility:"      "$N_FEASIBLE"  "$N_INFEASIBLE" "$N_ERROR"
printf "  %-24s %s\n"  "Best cost:"             "$BEST_COST"
printf "  %-24s %s\n"  "Worst cost:"            "$WORST_COST"
printf "  %-24s %s\n"  "Avg cost:"              "$AVG_COST"
printf "  %-24s %s\n"  "Std dev cost:"          "$STD_COST"
printf "  %-24s %s s\n" "Avg time-to-best:"     "$AVG_TIME"
printf "  %-24s %s s\n" "Best time-to-best:"    "$BEST_TIME"
echo ""

# ─── Ghi stats.txt ────────────────────────────────────────────────────────────
STATS_FILE="${RESULT_DIR}/stats.txt"
{
    echo "Instance:            $INSTANCE_NAME"
    echo "Instance path:       $INSTANCE_PATH"
    echo "Num runs:            $NUM_RUNS"
    echo "Time limit (s):      $MAX_TIME"
    echo "Max iterations:      $MAX_ITER"
    echo "Date:                $(date '+%Y-%m-%d %H:%M:%S')"
    echo "---"
    echo "Valid runs:          $VALID_RUNS / $NUM_RUNS"
    echo "Feasible runs:       $N_FEASIBLE"
    echo "Infeasible runs:     $N_INFEASIBLE"
    echo "Error runs:          $N_ERROR"
    echo "---"
    echo "Best cost:           $BEST_COST"
    echo "Worst cost:          $WORST_COST"
    echo "Avg cost:            $AVG_COST"
    echo "Std dev cost:        $STD_COST"
    echo "---"
    echo "Avg time-to-best:    $AVG_TIME s"
    echo "Best time-to-best:   $BEST_TIME s"
} > "$STATS_FILE"

echo -e "${GREEN}✔  Kết quả đã lưu vào: ${RESULT_DIR}/${RESET}"
echo -e "   summary.csv — kết quả từng lần chạy"
echo -e "   stats.txt   — thống kê tổng hợp"
echo -e "   run_N.log   — stdout (Final cost, feasibility)"
echo -e "   run_N.err   — stderr (ITER logs, time-to-best)"
echo ""
