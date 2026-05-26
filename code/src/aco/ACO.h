#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// FILE: ACO.h
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <cassert>
#include <vector>
#include <functional>
#include <chrono>
#include <fstream>
#include <string>
#include <queue>
#include <sstream>
#include <locale>

#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads() { return 1; }
#endif

using namespace std;

// ─────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────
using vecDbl  = vector<double>;
using matDbl  = vector<vector<double>>;
using Tengine = mt19937_64;

// ─────────────────────────────────────────────────────────────────────────
// Forward declaration: Instance và Parameters được định nghĩa trong Input.h.
// ACO.h chỉ cần biết chúng tồn tại để khai báo chữ ký ACO_tuned().
// ─────────────────────────────────────────────────────────────────────────
struct Instance;
struct Parameters;

// ═══════════════════════════════════════════════════════════════════════════
// ACOSolution — nghiệm của một con kiến (hoặc nghiệm tốt nhất toàn cục)
//
// assign[i]           = cluster mà node i thuộc về (0-indexed, 0 ≤ assign[i] < K)
// members[k]          = danh sách node thuộc cluster k
// clusterWeight[k][t] = tổng trọng số chiều t của cluster k
// clusterSumDist[i][k]= Σ dist(i,j) với j thuộc cluster k (maintained online)
// cost                = giá trị hàm mục tiêu (intra-dist + penalty)
// feasible            = true nếu tất cả ràng buộc trọng số thỏa mãn
// ═══════════════════════════════════════════════════════════════════════════
struct ACOSolution {
    double cost   = 1e300;
    bool feasible = false;

    vector<int>            assign;          // kích thước N
    vector<vector<int>>    members;         // kích thước K
    vector<vector<double>> clusterWeight;   // kích thước K × M_weights
    vector<vector<double>> clusterSumDist;  // kích thước N × K
};

// ═══════════════════════════════════════════════════════════════════════════
// LogRow — snapshot trạng thái mỗi 10 iteration, ghi vào file evolution
// ═══════════════════════════════════════════════════════════════════════════
struct LogRow {
    int    iter;            // số thứ tự iteration
    double time;            // thời gian trôi qua (giây)
    double bestCost;        // cost tốt nhất toàn cục tại thời điểm này
    bool   bestFeasible;    // best global có feasible không?
    double bestThisIter;    // cost tốt nhất trong iteration này
    int    feasibleAnts;    // số con kiến cho nghiệm feasible
    int    noImprove;       // số iteration liên tiếp không cải thiện
};

// ─────────────────────────────────────────────────────────────────────────
// Biến toàn cục của engine ACO
// (định nghĩa trong ACO.cpp, dùng chung bởi local_search, tabu_search,...)
// ─────────────────────────────────────────────────────────────────────────

// ── Logging ──
extern vector<LogRow> log_rows;
extern string         LOG_EVOL_FILENAME;    // đường dẫn file evolution
extern string         LOG_COST_FILENAME;    // đường dẫn file objectives
extern string         LOG_SOLU_FILENAME;    // đường dẫn file solutions

// ── Dữ liệu bài toán (được copy từ Instance vào khi khởi động ACO) ──
extern int    N;            // số node
extern int    K;            // số cluster
extern int    M_weights;    // số chiều trọng số

extern matDbl Wmat;         // Wmat[i][t]   = trọng số node i chiều t          (N × M_weights)
extern matDbl WLmat;        // WLmat[k][t]  = lower bound cluster k chiều t    (K × M_weights)
extern matDbl WUmat;        // WUmat[k][t]  = upper bound cluster k chiều t    (K × M_weights)
extern matDbl distmat;      // distmat[i][j]= khoảng cách i→j                  (N × N)

extern vector<vector<int>> globalCL;    // candidate list toàn cục (N × GLOBAL_CL_SIZE)

// ── Hằng số tính toán ──
extern double PENALTY_SCALE;    // hệ số phạt vi phạm (tính tự động theo instance)
extern double VALID_EPS;        // sai số kiểm tra so sánh

// ─────────────────────────────────────────────────────────────────────────
// Hàm công khai của ACO.cpp
// ─────────────────────────────────────────────────────────────────────────

// Tính cost đầy đủ từ vector assign (O(N²) — dùng để verify)
double compute_cost(const vector<int> &assign);

// Tính cost nhanh từ clusterSumDist đã maintain (O(N + K·M))
double compute_cost_fast(const ACOSolution &sol);

// Kiểm tra feasibility (ràng buộc trọng số, cluster id hợp lệ)
bool is_feasible(const vector<int> &assign);

// Đo khoảng cách Hamming giữa 2 nghiệm (đếm số node gán khác cluster)
int hamming_distance(const vector<int> &a, const vector<int> &b);

// Ghi 3 file log (evolution, objectives, solutions) ra đĩa
void SaveLogs(const ACOSolution &best);

// Hàm chính — chạy thuật toán ACO và trả nghiệm tốt nhất
// rng: engine được truyền từ ngoài (vengine[0] từ main.cpp)
//      → seed do main kiểm soát hoàn toàn, kết quả tái lập được khi dùng --seed
ACOSolution ACO_tuned(const Instance  &instance,
                      Tengine         &rng,
                      int              maxIter          = 10000,
                      double           timeLimitSeconds = 1200.0,
                      const string    &instance_name    = "");
