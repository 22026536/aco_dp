// ═══════════════════════════════════════════════════════════════════════════
// FILE: ACO.h
//
// Header chính của module ACO (Ant Colony Optimization).
// Khai báo tất cả các kiểu dữ liệu, cấu trúc, biến toàn cục, và prototype
// hàm được dùng chung giữa ACO.cpp, Local_search.cpp, Tabu_search.cpp
// và main.cpp.
//
// Nội dung:
//   1. Thư viện chuẩn
//   2. Hỗ trợ OpenMP (tuỳ chọn)
//   3. Type aliases        (vecDbl, matDbl, Tengine)
//   4. Struct Parameters   — cấu hình chạy chương trình
//   5. Struct Instance     — dữ liệu bài toán đầu vào
//   6. Struct ACOSolution  — nghiệm của một con kiến
//   7. Struct LogRow       — một snapshot trong log tiến hóa
//   8. Biến toàn cục extern — dữ liệu bài toán, tham số, đường dẫn log
//   9. SplitString()       — hàm tiện ích inline
//  10. Khai báo hàm
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

// ─────────────────────────────────────────────────────────────────────────
// 1. Thư viện chuẩn
// ─────────────────────────────────────────────────────────────────────────
#include <iostream>     // cout, cerr
#include <iomanip>      // setw, setprecision, fixed
#include <algorithm>    // sort, shuffle, min, max, find, any_of, ...
#include <cmath>        // sqrt, pow, round, log, exp, abs
#include <random>       // mt19937_64, uniform_real_distribution, random_device
#include <numeric>      // iota, accumulate
#include <cassert>      // assert()
#include <vector>       // std::vector
#include <functional>   // std::function
#include <chrono>       // steady_clock, duration_cast
#include <fstream>      // ifstream, ofstream
#include <string>       // std::string
#include <queue>        // std::priority_queue
#include <sstream>      // std::stringstream, std::ostringstream
#include <locale>       // std::locale — định dạng số có dấu phân nghìn

// ─────────────────────────────────────────────────────────────────────────
// 2. OpenMP (hỗ trợ song song, tuỳ chọn)
//
// Nếu biên dịch với -fopenmp → include header OpenMP thật sự.
// Nếu không → định nghĩa stub trả về 1 để các call site vẫn biên dịch được.
// ─────────────────────────────────────────────────────────────────────────
#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads() { return 1; }
#endif

using namespace std;

// ─────────────────────────────────────────────────────────────────────────
// 3. Type aliases
// ─────────────────────────────────────────────────────────────────────────
using vecDbl  = vector<double>;          // mảng 1 chiều số thực
using matDbl  = vector<vector<double>>;  // ma trận 2 chiều số thực
using Tengine = mt19937_64;              // bộ sinh số ngẫu nhiên Mersenne Twister 64-bit

// ═══════════════════════════════════════════════════════════════════════════
// 4. Parameters — cấu hình chạy chương trình
//
// Tất cả các trường có giá trị mặc định hợp lý để chương trình có thể chạy
// ngay cả khi không truyền đủ tham số từ command line.
// Được điền bởi LoadInput() trong input.h.
// ═══════════════════════════════════════════════════════════════════════════
struct Parameters {

    // Tiêu chí dừng
    string ALGtc = "time";   // "time" → dừng sau ALGtv giây
                             // "iter" → dừng sau ALGtv vòng lặp
    double ALGtv = 1200.0;  // Giá trị tương ứng (mặc định 20 phút)

    // Ghi log
    string LOGdir = "results/logs";  // Thư mục gốc chứa các file output
    int    ALGlg  = 10;              // In một dòng log mỗi ALGlg iteration

    // Nhãn schema xây dựng nghiệm (giữ để tương thích với framework)
    string CONm = "ACO";

    // Tham số GRASP cũ (giữ để tương thích với script, không dùng trong ACO)
    string GRASPv = "";
    double GRASPa = 0.3;
    int    GRASPm = 1;
    int    GRASPb = 1;
    int    GRASPd = 1;

    // Tham số Local Search cũ (giữ để tương thích với script)
    string LSm = "relocate";  // loại move: "relocate" | "swap"
    bool   LSe = true;        // chế độ efficient (candidate list)
    string LSs = "best";      // chiến lược duyệt: "best" | "first" | "hybrid"

    // Chế độ debug
    bool DEBUG = false;       // nếu true → in thêm thông tin chẩn đoán chi tiết
};

// ═══════════════════════════════════════════════════════════════════════════
// 5. Instance — dữ liệu bài toán đầu vào
//
// Biểu diễn một instance MCGP (Multi-Constrained Graph Partitioning):
//   Cho đồ thị N đỉnh, phân hoạch thành K cluster sao cho tổng khoảng cách
//   intra-cluster nhỏ nhất, thoả mãn ràng buộc trọng số của từng cluster.
//
// Hỗ trợ 3 định dạng file: "p" (planar), "t" (tsp-like), "h" (handover).
// ═══════════════════════════════════════════════════════════════════════════
struct Instance {

    // Loại instance
    string type = "p";   // "p" — ma trận khoảng cách cho trực tiếp
                         // "t" — toạ độ (x, y); khoảng cách tính từ Euclidean
                         // "h" — mạng không dây (handover)

    // Kích thước bài toán
    int nV = 0;   // số đỉnh (nodes/vertices)
    int nK = 0;   // số cluster (partitions)
    int nT = 0;   // số chiều trọng số (loại tài nguyên)

    // Dữ liệu trọng số và khoảng cách
    matDbl W;    // W[i][t]  = trọng số node i chiều t              (nV × nT)
    matDbl WL;   // WL[k][t] = giới hạn dưới cluster k chiều t     (nK × nT)
    matDbl WU;   // WU[k][t] = giới hạn trên cluster k chiều t     (nK × nT)
    matDbl D;    // D[i][j]  = khoảng cách từ node i đến node j    (nV × nV)
                 // Có thể không đối xứng. Mục tiêu: min Σ_{i<j, cùng cluster} (D[i][j]+D[j][i])
};

// ═══════════════════════════════════════════════════════════════════════════
// 6. ACOSolution — nghiệm của một con kiến
//
// Được xây dựng bởi một con kiến, sau đó cải thiện qua Local Search và
// Tabu Search trước khi dùng để cập nhật ma trận pheromone.
//
// Để tính cost trong O(N) thay vì O(N²), hai cấu trúc tăng dần được duy trì:
//   clusterWeight   — tổng trọng số hiện tại theo (cluster, chiều)
//   clusterSumDist  — tổng khoảng cách từ mỗi node đến từng cluster
// Cả hai được cập nhật tăng dần khi gán node vào cluster.
// ═══════════════════════════════════════════════════════════════════════════
struct ACOSolution {

    double cost   = 1e300;   // Giá trị hàm mục tiêu: intra_dist + PENALTY_SCALE × vi_phạm
                             // Khởi tạo ∞ để mọi nghiệm thật đều tốt hơn.
    bool feasible = false;   // true khi tất cả ràng buộc trọng số được thoả mãn

    vector<int>          assign;   // assign[i] = chỉ số cluster (0-based) của node i  (kích thước N)
    vector<vector<int>>  members;  // members[k] = danh sách node thuộc cluster k       (kích thước K)

    vector<vector<double>> clusterWeight;
        // clusterWeight[k][t] = tổng trọng số hiện tại của cluster k chiều t
        // Cập nhật tăng dần khi xây dựng nghiệm; dùng để kiểm tra feasibility.

    vector<vector<double>> clusterSumDist;
        // clusterSumDist[i][k] = Σ D[i][j] với mọi j hiện có trong cluster k
        // Cập nhật O(N) khi thêm node; giúp compute_cost_fast() chạy trong O(N).
};

// ═══════════════════════════════════════════════════════════════════════════
// 7. LogRow — một snapshot trong log tiến hóa
//
// Được thêm vào mỗi ALGlg iteration; ghi ra file evolution khi kết thúc.
// ═══════════════════════════════════════════════════════════════════════════
struct LogRow {
    int    iter;           // số thứ tự iteration hiện tại
    double time;           // thời gian đã trôi qua kể từ khi bắt đầu (giây)
    double bestCost;       // cost tốt nhất toàn cục tại snapshot này
    bool   bestFeasible;   // nghiệm tốt nhất toàn cục có feasible không
    double bestThisIter;   // cost ant tốt nhất trong iteration này (trước khi so với global best)
    int    feasibleAnts;   // số ant tạo ra nghiệm feasible trong iteration này
    int    noImprove;      // số iteration liên tiếp không cải thiện được global best
};

// ─────────────────────────────────────────────────────────────────────────
// 8. Biến toàn cục extern
//
// Được định nghĩa trong ACO.cpp; khai báo ở đây để các translation unit
// khác truy cập được mà không cần truyền qua tham số hàm.
// ─────────────────────────────────────────────────────────────────────────

// Buffer log
extern vector<int>    log_iter;   // danh sách iteration đã log (phụ trợ)
extern vector<double> log_time;   // danh sách timestamp đã log (phụ trợ)
extern vector<LogRow> log_rows;   // snapshot đầy đủ, ghi ra file khi kết thúc

// Cấu hình chạy (được điền bởi LoadInput trong main.cpp)
extern Parameters parameters;

// Đường dẫn file log (được set bởi ACO_tuned trước lần ghi đầu tiên)
extern string LOG_EVOL_FILENAME;  // file lịch sử hội tụ
extern string LOG_COST_FILENAME;  // file cost tốt nhất (1 dòng)
extern string LOG_SOLU_FILENAME;  // file nghiệm tốt nhất (danh sách node mỗi cluster)

// Kích thước bài toán (copy từ Instance bên trong ACO_tuned)
extern int N;          // số node
extern int K;          // số cluster
extern int M_weights;  // số chiều trọng số

// Ma trận dữ liệu bài toán (set một lần trong ACO_tuned, sau đó chỉ đọc)
extern matDbl Wmat;    // Wmat[i][t]  = trọng số node           (N × M_weights)
extern matDbl WLmat;   // WLmat[k][t] = giới hạn dưới cluster   (K × M_weights)
extern matDbl WUmat;   // WUmat[k][t] = giới hạn trên cluster   (K × M_weights)
extern matDbl distmat; // distmat[i][j] = khoảng cách i → j     (N × N)

// Hằng số điều chỉnh (tính tự động từ đặc tính instance)
extern double PENALTY_SCALE;
    // Hệ số nhân cho vi phạm ràng buộc trong hàm mục tiêu:
    //   cost = intra_distance + PENALTY_SCALE × tổng_vi_phạm
    // Được hiệu chỉnh để một đơn vị vi phạm xấp xỉ bằng một đơn vị khoảng cách,
    // cân bằng giữa feasibility và chất lượng nghiệm.

extern double VALID_EPS;
    // Ngưỡng epsilon tương đối cho kiểm tra feasibility.
    // Một ràng buộc được coi là thoả mãn khi:
    //   |tổng_trọng_số - bound| < VALID_EPS × scale
    // Tránh lỗi làm tròn dấu phẩy động làm từ chối nghiệm hợp lệ.

// ═══════════════════════════════════════════════════════════════════════════
// 9. SplitString() — hàm tiện ích tách chuỗi (inline)
//
// Tách một chuỗi thành các token dựa trên ký tự phân cách.
// Ví dụ: SplitString("1 23 456", ' ') → {"1", "23", "456"}
// Dùng trong LoadInstance() khi đọc file instance loại "h".
// ═══════════════════════════════════════════════════════════════════════════
inline vector<string> SplitString(const string &str, char delimiter)
{
    vector<string> tokens;
    stringstream   ss(str);
    string         token;
    while (getline(ss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Khai báo hàm
// ═══════════════════════════════════════════════════════════════════════════

// Hàm nội bộ — khai báo ở đây để các translation unit include ACO.h
// thấy được signature, dù thân hàm nằm trong ACO.cpp.
static inline string format_cost_with_commas(double v, int decimals = 0);

// ── Hàm tính chi phí ─────────────────────────────────────────────────────

// Tính lại cost đầy đủ O(N²) từ vector assign.
// Dùng để kiểm tra kết quả hoặc khi không có dữ liệu tăng dần.
double compute_cost(const vector<int> &assign);

// Tính cost nhanh O(N + K·M) từ cấu trúc clusterSumDist đã duy trì.
// Dùng trong vòng lặp chính để tránh tính O(N²) mỗi lần đánh giá nghiệm.
double compute_cost_fast(const ACOSolution &sol);

// ── Hàm kiểm tra ─────────────────────────────────────────────────────────

// Trả về true nếu tất cả ràng buộc trọng số được thoả mãn trong VALID_EPS.
bool is_feasible(const vector<int> &assign);

// Đếm số node được gán vào cluster khác nhau giữa nghiệm a và b.
// Dùng để đo độ đa dạng (diversity) giữa các ant trong population.
int hamming_distance(const vector<int> &a, const vector<int> &b);

// ── Hàm ghi log ──────────────────────────────────────────────────────────

// Ghi 3 file output sau khi thuật toán kết thúc:
//   (1) Evolution log  — snapshot cost qua các iteration
//   (2) Objectives log — giá trị cost tốt nhất (1 dòng)
//   (3) Solution log   — danh sách node trong mỗi cluster (chỉ số 1-based)
void SaveLogs(const ACOSolution &best);

// ── Thuật toán chính ──────────────────────────────────────────────────────

// ACO_tuned() — Ant Colony Optimization cho bài toán MCGP.
//
// Tham số:
//   instance          : dữ liệu bài toán (N, K, M_weights, W, WL, WU, D)
//   maxIter           : giới hạn số iteration (mặc định 10 000)
//   timeLimitSeconds  : giới hạn thời gian thực (giây, mặc định 1 200)
//   instance_name     : dùng để đặt tên các file log output
//
// Trả về nghiệm ACOSolution tốt nhất tìm được trong ngân sách cho phép.
ACOSolution ACO_tuned(const Instance &instance,
                      int    maxIter          = 10000,
                      double timeLimitSeconds = 1200.0,
                      const  string &instance_name = "");
