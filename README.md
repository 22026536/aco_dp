# ACO-MCGP

Giải bài toán **MCGP** (Multi-Constrained Graph Partitioning — Phân cụm đồ thị có ràng buộc trọng số) bằng **Ant Colony Optimization** kết hợp **Local Search** và **Tabu Search**.

---

## Mục lục

- [Bài toán](#bài-toán)
- [Thuật toán](#thuật-toán)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & Biên dịch](#cài-đặt--biên-dịch)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cách chạy](#cách-chạy)
  - [Chạy một lần](#1-chạy-một-lần--run-acosh)
  - [Chạy toàn bộ thư mục](#2-chạy-toàn-bộ-thư-mục--auto-runsh)
  - [Benchmark một instance](#3-benchmark-một-instance--run-benchmarksh)
  - [Benchmark toàn bộ thư mục](#4-benchmark-toàn-bộ-thư-mục--auto-run-benchmarksh)
  - [Chạy thủ công (binary trực tiếp)](#5-chạy-thủ-công-binary-trực-tiếp)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Định dạng file instance](#định-dạng-file-instance)
- [Tham số dòng lệnh](#tham-số-dòng-lệnh)

---

## Bài toán

Cho một đồ thị có **N node**, cần chia thành **K cluster** sao cho:

- **Mục tiêu:** Tổng khoảng cách intra-cluster (giữa các node trong cùng cluster) nhỏ nhất.
- **Ràng buộc:** Mỗi cluster k phải thỏa mãn với mỗi chiều trọng số t:
  ```
  WL[k][t] ≤ tổng_trọng_số_cluster_k_chiều_t ≤ WU[k][t]
  ```

---

## Thuật toán

```
Vòng lặp ACO (mỗi iteration):
  ├── Construction: m con kiến xây nghiệm dựa trên pheromone + heuristic
  ├── Local Search: cải thiện top ants bằng local search
  ├── Tabu Search:  thoát local optima của local search bằng tabu search (chạy mỗi tsInterval iteration)
  └── Pheromone update: evaporation + deposit từ best solution
```

Nếu không cải thiện sau 30 iteration liên tiếp → **reset pheromone** để thoát stagnation.

---

## Cài đặt & Biên dịch

### Bước 1: Clone repository

```bash
git clone <repo_url>
cd <repo_name>
```

### Bước 2: Biên dịch
Nếu chưa có file MCGP trong thư mục hiện tại, cần biên dịch , ví dụ trong WSL
```bash
mkdir -p code/build
cd code/build
cmake ..
make -j$(nproc)
cd ../..
```

Sau khi biên dịch thành công, binary `MCGP` sẽ xuất hiện trong **thư mục gốc** (nơi chứa các script `.sh`).

> **Lưu ý:** Tất cả các script `.sh` đều kỳ vọng binary `./MCGP` nằm trong thư mục hiện tại (thư mục gốc dự án). Đừng chạy script từ trong `code/build/`.

### Cấp quyền thực thi cho các script

```bash
chmod +x run-aco.sh run-benchmark.sh auto-run.sh auto-run-benchmark.sh
```

---

## Cấu trúc thư mục

```
.
├── MCGP                        ← Binary sau khi biên dịch
├── run-aco.sh                  ← Chạy ACO một lần
├── run-benchmark.sh            ← Benchmark một instance nhiều lần
├── auto-run.sh                 ← Chạy ACO cho toàn bộ thư mục instance
├── auto-run-benchmark.sh       ← Benchmark toàn bộ thư mục instance
├── compile.sh                  ← Script biên dịch nhanh (không dùng -j)
├── code/                       ← Source code C++
│   ├── CMakeLists.txt
│   ├── aco/
│   │   ├── ACO.cpp / ACO.h
│   │   ├── Input.cpp / Input.h
│   │   ├── Local_search.cpp / Local_search.h
│   │   └── Tabu_search.cpp / Tabu_search.h
│   └── main.cpp
├── normalized_instances/       ← Thư mục dữ liệu instance (mặc định)
│   └── tsplib/
│       ├── inst_100_5.txt
│       └── ...
└── results/                    ← Kết quả đầu ra (tự động tạo khi chạy)
    ├── logs/                   ← Kết quả từ run-aco.sh / auto-run.sh
    │   └── <instance_name>/
    │       ├── evolution/      ← Diễn biến hội tụ qua các iteration
    │       ├── solutions/      ← Nghiệm tốt nhất (danh sách node mỗi cluster)
    │       └── objectives/     ← Cost tốt nhất (1 con số)
    ├── summary/                ← Kết quả từ auto-run.sh
    │   └── <instance_name>/logs/
    └── benchmark/              ← Kết quả từ run-benchmark.sh
        ├── <instance_name>/
        │   ├── run_1.log       ← stdout lần chạy 1
        │   ├── run_1.err       ← stderr lần chạy 1 (ITER logs, time-to-best)
        │   ├── run_2.log
        │   ├── run_2.err
        │   ├── summary.csv     ← Bảng kết quả từng lần chạy
        │   └── stats.txt       ← Thống kê tổng hợp (avg/best/std)
        └── _summary_all.csv    ← Tổng hợp tất cả instance (từ auto-run-benchmark.sh)
```

---

## Cách chạy

### 1. Chạy một lần — `run-aco.sh`

Chạy ACO **một lần** trên một instance, với seed ngẫu nhiên.

```bash
./run-aco.sh <instance_path> [max_time_giây] [max_iter]
```

**Ví dụ:**
```bash
./run-aco.sh normalized_instances/tsplib/kroA100_05_5.txt 300 5000
```

**Tham số:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `instance_path` | *(bắt buộc)* | Đường dẫn tới file instance |
| `max_time` | `1500` | Giới hạn thời gian chạy (giây) |
| `max_iter` | `10000` | Số iteration tối đa |

**Kết quả** được lưu vào:
```
results/logs/<instance_name>/evolution/<instance_name>    ← diễn biến hội tụ
results/logs/<instance_name>/solutions/<instance_name>    ← nghiệm tốt nhất
results/logs/<instance_name>/objectives/<instance_name>   ← cost tốt nhất
```

---

### 2. Chạy toàn bộ thư mục — `auto-run.sh`

Duyệt **tất cả file `.txt`** trong `normalized_instances/tsplib/` và chạy ACO lần lượt. Time limit được chọn **tự động** theo kích thước instance (dựa vào số đầu tiên trong tên file).

```bash
./auto-run.sh
```

Bảng time limit tự động:

| Kích thước N | Time limit |
|---|---|
| 1 – 99 | 10 giây |
| 100 – 199 | 60 giây |
| 200 – 399 | 300 giây |
| ≥ 400 | 1500 giây |

**Kết quả** được copy sang:
```
results/summary/<instance_name>/logs/
```

---

### 3. Benchmark một instance — `run-benchmark.sh`

Chạy **nhiều lần** với các seed khác nhau để đánh giá độ ổn định của thuật toán.

```bash
./run-benchmark.sh <instance_path> [num_runs] [max_time] [max_iter]
```

**Ví dụ:**
```bash
./run-benchmark.sh normalized_instances/tsplib/kroA100_05_5.txt 10 60 1000
```

**Tham số:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `instance_path` | *(bắt buộc)* | Đường dẫn tới file instance |
| `num_runs` | `10` | Số lần chạy |
| `max_time` | `1500` | Giới hạn thời gian mỗi lần (giây) |
| `max_iter` | `10000` | Số iteration tối đa mỗi lần |

**Kết quả** được lưu vào `results/benchmark/<instance_name>/`:

| File | Nội dung |
|---|---|
| `run_N.log` | stdout lần chạy thứ N (Final cost, feasibility) |
| `run_N.err` | stderr lần chạy thứ N (ITER logs, time-to-best) |
| `summary.csv` | Bảng: run, seed, feasible, best_cost, time_to_best |
| `stats.txt` | Thống kê tổng hợp: avg/best/worst/std cost, avg time-to-best |

**Ví dụ nội dung `stats.txt`:**
```
Instance:            inst_100_5
Num runs:            10
Time limit (s):      300
Valid runs:          10 / 10
Feasible runs:       9
Best cost:           123456.78
Avg cost:            125000.00
Std dev cost:        1200.50
Avg time-to-best:    45.32 s
```

---

### 4. Benchmark toàn bộ thư mục — `auto-run-benchmark.sh`

Chạy `run-benchmark.sh` lần lượt trên **tất cả instance** trong một thư mục, rồi tổng hợp kết quả vào một file CSV duy nhất.

```bash
./auto-run-benchmark.sh <instance_dir> [num_runs] [time_limit]
```

**Ví dụ:**
```bash
# Time limit tự động theo kích thước
./auto-run-benchmark.sh normalized_instances/tsplib 10

# Time limit cố định 300s cho tất cả
./auto-run-benchmark.sh normalized_instances/tsplib 10 300
```

**Kết quả:**
- Mỗi instance → `results/benchmark/<instance_name>/` (giống mục 3)
- Tổng hợp tất cả → `results/benchmark/_summary_all.csv`

**Định dạng `_summary_all.csv`:**
```
instance,n,feasible_runs,best_cost,avg_cost,std_cost,avg_time_to_best_s,time_limit_s
inst_100_5,100,10,123456.78,125000.00,1200.50,45.32,300
inst_200_8,200,8,456789.12,...
```

---

### 5. Chạy thủ công (binary trực tiếp)

```bash
./MCGP \
  --instance          <path_to_instance> \
  --termination_value <time_limit_giây>  \
  --iter_value        <max_iterations>   \
  --seed              <integer>          \
  --logs              10
```

Thuật toán sẽ dừng khi đạt giới hạn thời gian hoặc số iteration tối đa, tùy điều kiện nào đến trước.

**Ví dụ:**
```bash
./MCGP \
  --instance          normalized_instances/tsplib/kroA100_05_5.txt \
  --termination_value 300  \
  --iter_value        5000 \
  --seed              42
```

---

## Kết quả đầu ra

### File `evolution/<instance_name>`

Diễn biến hội tụ qua các iteration (ghi mỗi 10 vòng):

```
# iter   time(s)    bestCost    bestFeasible  bestThisIter  feasibleAnts  noImprove
    10     12.3400  123456.78         1       124000.00            35         0
    20     24.1200  122000.50         1       123000.00            38         0
   ...
```

### File `objectives/<instance_name>`

Cost tốt nhất (một số duy nhất):
```
122000.500000
```

### File `solutions/<instance_name>`

Nghiệm tốt nhất: mỗi dòng là một cluster, liệt kê các node (đánh số từ 1) cách nhau bằng dấu cách:
```
1 5 12 34 67
2 8 15 23 56
3 9 18 41 72
```

---

## Định dạng file instance

```
N K T
W[0][0] W[0][1] ... W[0][T-1]
W[1][0] ...
...
W[N-1][0] ... W[N-1][T-1]
WL[0][0] WU[0][0] WL[0][1] WU[0][1] ...    ← bounds cluster 0
WL[1][0] WU[1][0] ...                       ← bounds cluster 1
...
WL[K-1][0] WU[K-1][0] ...
D[0][0] D[0][1] ... D[0][N-1]
D[1][0] ...
...
D[N-1][0] ... D[N-1][N-1]
```

Trong đó:
- `N` = số node, `K` = số cluster, `T` = số chiều trọng số
- `W[i][t]` = trọng số node i chiều t
- `WL[k][t]`, `WU[k][t]` = lower/upper bound cluster k chiều t
- `D[i][j]` = khoảng cách từ node i đến node j (ma trận có thể không đối xứng)

---

## Tham số dòng lệnh

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `--instance` | ✓ | — | Đường dẫn file instance |
| `--termination_value` | ✓ | — | Giới hạn thời gian chạy (giây) |
| `--iter_value` | ✓ | — | Số iteration tối đa |
| `--seed` | | 0 (ngẫu nhiên) | Seed cho bộ sinh số ngẫu nhiên. `0` = random mỗi lần |
| `--logs` | | 10 | Ghi log mỗi bao nhiêu iteration |
