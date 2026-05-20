# ACO-DP

Giải bài toán **MCGP** (phân cụm có ràng buộc trọng số) bằng Ant Colony Optimization + Local Search + Tabu Search.

## Yêu cầu

Binary `./MCGP` đã được compile trong thư mục gốc.

## Scripts

### `run-aco.sh` — Chạy ACO một lần
```bash
./run-aco.sh <instance_path> [time_limit]
```
Kết quả: `results/logs/<instance_name>/{evolution,solutions,objectives}/`

---

### `auto_run.sh` — Chạy ACO toàn bộ thư mục
```bash
./auto_run.sh
```
Duyệt tất cả `.txt` trong `normalized_instances/tsplib/`, time limit tự động theo tên file.  
Kết quả: `results/summary/<instance_name>/logs/`

---

### `run-benchmark.sh` — Benchmark một instance
```bash
./run-benchmark.sh <instance_path> [num_runs=10] [time_limit=1200]
```
Chạy nhiều lần với seed khác nhau, tính avg/best/std cost và time-to-best.  
Kết quả: `results/benchmark/<instance_name>/{run_N.log, run_N.err, summary.csv, stats.txt}`

---

### `auto-run-benchmark.sh` — Benchmark toàn bộ thư mục
```bash
./auto-run-benchmark.sh <instance_dir> [num_runs=10] [time_limit=tự động]
```
Kết quả: `results/benchmark/<instance_name>/` + file tổng hợp `results/benchmark/_summary_all.csv`
