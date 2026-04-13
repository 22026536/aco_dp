// ═══════════════════════════════════════════════════════════════════════════
// FILE: ACO.cpp
//
// BÀI TOÁN: MCGP (Multi-Constrained Graph Partitioning)
//   - Cho N node, K cluster, M_weights chiều trọng số
//   - Mỗi node i có trọng số Wmat[i][t] trên từng chiều t
//   - Mỗi cluster k có lower bound WLmat[k][t] và upper bound WUmat[k][t]
//   - Ma trận khoảng cách distmat[i][j] giữa các cặp node
//   - MỤC TIÊU: Gán mỗi node vào đúng 1 cluster sao cho:
//       (1) Tổng khoảng cách intra-cluster nhỏ nhất
//       (2) Ràng buộc trọng số: WLmat[k][t] <= tổng_W_cluster_k_t <= WUmat[k][t]
//
// THUẬT TOÁN: Ant Colony Optimization (ACO)
//   - Mỗi iteration: m con kiến xây nghiệm dựa trên pheromone + heuristic
//   - Top ants được cải thiện bằng Local Search + Tabu Search
//   - Pheromone được cập nhật dựa trên nghiệm tốt nhất
//   - Lặp cho đến khi hết thời gian hoặc đạt maxIter
// ═══════════════════════════════════════════════════════════════════════════

#include "ACO.h"            // Header chứa khai báo struct ACOSolution, Instance, Parameters, LogRow,...
#include "Local_search.h"   // Header cho hàm local_search() — cải thiện nghiệm bằng relocate/swap
#include "Large_search.h"   // Header cho large neighborhood search (nếu dùng)
#include "Tabu_search.h"    // Header cho iterated_tabu_search() — tìm kiếm tabu lặp
#include <iostream>         // cout, cerr — in ra console
#include <iomanip>          // setw, setprecision — định dạng output
#include <algorithm>        // sort, shuffle, min, max, find, any_of,...
#include <cmath>            // sqrt, pow, exp, abs — hàm toán học
#include <random>           // mt19937_64, uniform_real_distribution — sinh số ngẫu nhiên
#include <numeric>          // iota (fill 0,1,2,...), accumulate (tính tổng)
#include <cassert>          // assert — kiểm tra điều kiện debug
#include <vector>           // vector — cấu trúc dữ liệu chính
#include <functional>       // function objects (không dùng trực tiếp ở đây)
#include <chrono>           // steady_clock, duration — đo thời gian chạy
#include <fstream>          // ofstream — ghi file log
#include <string>           // string — chuỗi ký tự
#include <queue>            // priority_queue (không dùng trực tiếp ở đây)
#include <sstream>          // ostringstream — format chuỗi
#include <locale>           // locale — hỗ trợ format số có dấu phẩy ngàn

using std::mt19937_64;      // alias cho Mersenne Twister 64-bit RNG
using std::vector;          // alias cho std::vector
using Clock = chrono::steady_clock;
                            // Clock dùng để đo thời gian chạy (monotonic, không bị ảnh hưởng bởi đổi giờ hệ thống)

// ═══════════════════════════════════════════════════════════════════════════
// BIẾN TOÀN CỤC (GLOBAL VARIABLES)
//
// Được khởi tạo trong ACO_tuned() và sử dụng bởi tất cả các module
// (local_search, tabu_search, compute_cost, is_feasible,...)
// ═══════════════════════════════════════════════════════════════════════════

vector<int> log_iter;           // [LOG] danh sách iteration đã log (dự phòng, chưa dùng trực tiếp)
vector<double> log_time;        // [LOG] danh sách thời gian tại mỗi log point (dự phòng)
vector<LogRow> log_rows;        // [LOG] danh sách snapshot mỗi 10 iteration
                                //   LogRow chứa: iter, time, bestCost, bestFeasible,
                                //                bestThisIter, feasibleAnts, noImprove

Parameters parameters;          // Tham số cấu hình (đọc từ file hoặc command line)
                                //   Chứa: LOGdir, timeLimitSeconds, maxIter, ...

string LOG_EVOL_FILENAME;       // Đường dẫn file log evolution (ghi diễn biến qua các iteration)
string LOG_COST_FILENAME;       // Đường dẫn file log cost tốt nhất
string LOG_SOLU_FILENAME;       // Đường dẫn file log nghiệm tốt nhất (danh sách cluster)

// ─── DỮ LIỆU BÀI TOÁN (được copy từ Instance vào biến toàn cục) ───

int N = 0;                      // Số lượng node trong đồ thị (vertices)
int K = 0;                      // Số lượng cluster (partitions) cần chia
int M_weights = 0;              // Số chiều trọng số (resource dimensions)
                                //   Ví dụ: M_weights=2 nghĩa là mỗi node có 2 loại trọng số
                                //   (ví dụ: dân số + diện tích)

vector<vector<double>> Wmat;    // Wmat[i][t] = trọng số của node i tại chiều t
                                //   Kích thước: N x M_weights
                                //   Ví dụ: Wmat[3][0] = 150 → node 3 có trọng số 150 ở chiều 0

vector<vector<double>> WLmat;   // WLmat[k][t] = lower bound (giới hạn dưới) của cluster k tại chiều t
                                //   Kích thước: K x M_weights
                                //   Ràng buộc: tổng W node trong cluster k >= WLmat[k][t]

vector<vector<double>> WUmat;   // WUmat[k][t] = upper bound (giới hạn trên) của cluster k tại chiều t
                                //   Kích thước: K x M_weights
                                //   Ràng buộc: tổng W node trong cluster k <= WUmat[k][t]

vector<vector<double>> distmat; // distmat[i][j] = khoảng cách từ node i đến node j
                                //   Kích thước: N x N
                                //   Có thể đối xứng (distmat[i][j] == distmat[j][i]) hoặc không
                                //   Đây là chi phí intra-cluster: ta muốn minimize tổng dist các cặp
                                //   node cùng cluster

double PENALTY_SCALE = 10000.0; // Hệ số phạt cho vi phạm ràng buộc trọng số
                                //   cost = intra_distance + PENALTY_SCALE * total_violation
                                //   Giá trị lớn → ưu tiên tìm nghiệm feasible trước
                                //   Sẽ được tính lại dựa trên instance trong ACO_tuned()

double VALID_EPS = 1e-6;        // Epsilon cho kiểm tra feasibility
                                //   Cho phép sai số nhỏ khi so sánh với bounds
                                //   Ví dụ: nếu tổng W = 99.9999999 và lower = 100, vẫn coi là OK


// ═══════════════════════════════════════════════════════════════════════════
// HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
// ═══════════════════════════════════════════════════════════════════════════

// ─── Format số thực dạng fixed (không dùng ký hiệu khoa học) ───
// Ví dụ: format_cost_fixed(12345.678, 2) → "12345.68"
// Ví dụ: format_cost_fixed(12345.678, 0) → "12346"
//
// v:        giá trị cần format
// decimals: số chữ số thập phân (0 → số nguyên)
// return:   chuỗi đã format
static inline std::string format_cost_fixed(double v, int decimals)
{
    std::ostringstream oss;                             // tạo string stream
    oss << std::fixed                                   // dùng fixed notation (không scientific)
        << std::setprecision(decimals)                  // đặt số chữ số sau dấu chấm
        << v;                                           // ghi giá trị vào stream
    return oss.str();                                   // trả về chuỗi kết quả
}

// ─── Format số với dấu phân cách hàng nghìn (locale-dependent) ───
// Ví dụ: format_cost_with_commas(1234567.89, 0) → "1,234,568" (tùy locale)
//
// Dùng std::locale("") để lấy locale hệ thống.
// Nếu hệ thống hỗ trợ → sẽ thêm dấu phẩy/dấu chấm phân cách nghìn.
// Nếu không hỗ trợ → trả về số bình thường (không có separator).
//
// v:        giá trị cần format
// decimals: số chữ số thập phân
// return:   chuỗi đã format
static inline std::string format_cost_with_commas(double v, int decimals)
{
    std::ostringstream oss;                             // tạo string stream
    try
    {
        oss.imbue(std::locale(""));                     // áp dụng locale hệ thống
                                                        // → cho phép grouping digits (1,000,000)
    }
    catch (...)
    {
        // Nếu locale không được hỗ trợ → bỏ qua, dùng locale mặc định
        // (không crash, chỉ mất dấu phân cách)
    }
    oss << std::fixed                                   // fixed notation
        << std::setprecision(decimals)                  // số decimal
        << v;                                           // giá trị
    return oss.str();                                   // trả chuỗi
}


// ═══════════════════════════════════════════════════════════════════════════
// KIỂM TRA TÍNH HỢP LỆ CỦA INSTANCE
//
// Kiểm tra: tổng trọng số tất cả node có nằm trong khoảng
//           [tổng lower bound, tổng upper bound] hay không?
//
// Nếu KHÔNG → instance không thể có nghiệm feasible → báo lỗi ngay.
//
// Ví dụ: 3 cluster, mỗi cluster cần ít nhất 100 đơn vị trọng số
//        → tổng lower = 300
//        Nếu tổng trọng số node = 250 → KHÔNG THỂ feasible → trả false
// ═══════════════════════════════════════════════════════════════════════════
bool check_weights_validity(const Instance instance)
{
    int N = instance.nV;                                // số node
    int K = instance.nK;                                // số cluster
    int T = instance.nT;                                // số chiều trọng số

    // Kiểm tra kích thước cơ bản
    if (N <= 0 || K <= 0 || T <= 0)
    {
        cerr << "[ERROR] Invalid sizes\n";              // in lỗi ra stderr
        return false;                                   // instance không hợp lệ
    }

    vector<double> sumNode(T, 0.0);                     // sumNode[t] = tổng trọng số chiều t của TẤT CẢ node
    vector<double> sumMin(T, 0.0);                      // sumMin[t] = tổng lower bound chiều t của TẤT CẢ cluster
    vector<double> sumMax(T, 0.0);                      // sumMax[t] = tổng upper bound chiều t của TẤT CẢ cluster

    // Tính tổng trọng số tất cả node, từng chiều
    for (int i = 0; i < N; ++i)                         // duyệt mỗi node
    {
        for (int t = 0; t < T; ++t)                     // duyệt mỗi chiều trọng số
            sumNode[t] += instance.W[i][t];             // cộng dồn trọng số node i chiều t
    }

    // Tính tổng lower/upper bound tất cả cluster, từng chiều
    for (int k = 0; k < K; ++k)                         // duyệt mỗi cluster
    {
        for (int t = 0; t < T; ++t)                     // duyệt mỗi chiều
        {
            sumMin[t] += instance.WL[k][t];             // cộng dồn lower bound cluster k chiều t
            sumMax[t] += instance.WU[k][t];             // cộng dồn upper bound cluster k chiều t
        }
    }

    // Kiểm tra: tổng node phải nằm trong [tổng min, tổng max] cho mỗi chiều
    for (int t = 0; t < T; ++t)
    {
        // sumNode[t] < sumMin[t] → thiếu trọng số, không đủ fill tất cả lower bound
        // sumNode[t] > sumMax[t] → quá nhiều trọng số, tràn tất cả upper bound
        if (sumNode[t] < sumMin[t] - 1e-9 || sumNode[t] > sumMax[t] + 1e-9)
        {
            cerr << "\n[INVALID] Weight type " << t << "\n";
            cerr << "  Sum node = " << sumNode[t] << "\n";    // tổng trọng số thực tế
            cerr << "  Sum min  = " << sumMin[t] << "\n";     // tổng lower bound
            cerr << "  Sum max  = " << sumMax[t] << "\n";     // tổng upper bound
            return false;                                      // instance không khả thi
        }
    }

    return true;                                        // tất cả chiều OK → instance hợp lệ
}


// ═══════════════════════════════════════════════════════════════════════════
// TÍNH CHI PHÍ (COST) CỦA MỘT NGHIỆM
//
// cost = intra_distance + PENALTY_SCALE * total_violation
//
//   intra_distance = tổng khoảng cách giữa tất cả cặp node CÙNG cluster
//   total_violation = tổng lượng vi phạm ràng buộc trọng số
//
// Nếu nghiệm feasible → total_violation = 0 → cost = intra_distance thuần
// Nếu infeasible → penalty rất lớn → cost bị đẩy lên cao
//
// assign[i] = cluster mà node i thuộc về (0-indexed)
// ═══════════════════════════════════════════════════════════════════════════
double compute_cost(const vector<int> &assign)
{
    // ─── Bước 1: Tính tổng khoảng cách intra-cluster ───
    // Duyệt tất cả cặp (i,j) với i<j, nếu cùng cluster thì cộng dist cả 2 chiều
    // (distmat[i][j] + distmat[j][i] vì ma trận có thể KHÔNG đối xứng)
    double intra = 0.0;                                 // tổng intra-cluster distance
    for (int i = 0; i < N; ++i)                         // duyệt node i = 0..N-1
        for (int j = i + 1; j < N; ++j)                 // duyệt node j = i+1..N-1 (tránh đếm đôi)
            if (assign[i] == assign[j])                  // nếu i và j cùng cluster
                intra += distmat[i][j] + distmat[j][i]; // cộng khoảng cách cả 2 chiều

    // ─── Bước 2: Kiểm tra cluster id hợp lệ ───
    // Nếu có node nào bị gán cluster ngoài [0, K-1] → trả cost cực lớn (nghiệm lỗi)
    for (int i = 0; i < N; ++i)
        if (assign[i] < 0 || assign[i] >= K)            // cluster id không hợp lệ
            return 1e300;                                // trả cost "vô cực" → nghiệm bị loại bỏ

    // ─── Bước 3: Tính tổng trọng số mỗi cluster, mỗi chiều ───
    // sumW[k][t] = tổng Wmat[i][t] cho tất cả node i thuộc cluster k
    vector<vector<double>> sumW(K, vector<double>(M_weights, 0.0));
    for (int i = 0; i < N; ++i)                         // duyệt mỗi node
        for (int t = 0; t < M_weights; ++t)              // duyệt mỗi chiều trọng số
            sumW[assign[i]][t] += Wmat[i][t];            // cộng trọng số node i vào cluster của nó

    // ─── Bước 4: Tính tổng vi phạm ràng buộc trọng số ───
    double total_violation = 0.0;                        // tích lũy tổng vi phạm

    for (int k = 0; k < K; ++k)                         // duyệt mỗi cluster
    {
        for (int t = 0; t < M_weights; ++t)              // duyệt mỗi chiều
        {
            double s = sumW[k][t];                       // tổng trọng số thực tế của cluster k chiều t
            double low = WLmat[k][t];                    // lower bound
            double high = WUmat[k][t];                   // upper bound

            if (s < low)                                 // vi phạm lower bound (thiếu trọng số)
                total_violation += (low - s);            // lượng thiếu = lower - thực tế

            if (s > high)                                // vi phạm upper bound (thừa trọng số)
                total_violation += (s - high);           // lượng thừa = thực tế - upper
        }
    }

    // ─── Bước 5: Tổng chi phí = distance + penalty ───
    double penalty = total_violation * PENALTY_SCALE;    // phạt = vi phạm * hệ số phạt

    return intra + penalty;                              // tổng cost (nhỏ hơn = tốt hơn)
}


// ═══════════════════════════════════════════════════════════════════════════
// KIỂM TRA TÍNH KHẢ THI (FEASIBILITY) CỦA MỘT NGHIỆM
//
// Nghiệm feasible khi và chỉ khi:
//   (1) Mỗi node được gán vào đúng 1 cluster hợp lệ [0, K-1]
//   (2) Tổng trọng số mỗi cluster nằm trong [WLmat, WUmat] (cho mỗi chiều)
//
// Khác với compute_cost: hàm này trả true/false, không tính distance.
// Dùng relative tolerance (VALID_EPS * scale) thay vì absolute tolerance.
//
// assign[i] = cluster của node i
// return: true nếu feasible, false nếu vi phạm
// ═══════════════════════════════════════════════════════════════════════════
bool is_feasible(const std::vector<int> &assign)
{
    // ─── Kiểm tra kích thước vector ───
    if ((int)assign.size() != N)                         // assign phải có đúng N phần tử
        return false;

    // ─── Kiểm tra dữ liệu toàn cục đã khởi tạo ───
    if ((int)Wmat.size() != N ||                         // Wmat phải có N dòng
        (int)WLmat.size() != K ||                        // WLmat phải có K dòng
        (int)WUmat.size() != K)                          // WUmat phải có K dòng
        return false;

    // ─── Trường hợp đặc biệt: không có chiều trọng số ───
    if (M_weights == 0)
        return true;                                     // không có ràng buộc → luôn feasible

    // ─── Kiểm tra thêm: vector không rỗng và đúng số cột ───
    if (Wmat.empty() || WLmat.empty() || WUmat.empty())
        return false;
    if ((int)Wmat[0].size() != M_weights ||
        (int)WLmat[0].size() != M_weights ||
        (int)WUmat[0].size() != M_weights)
        return false;

    // ─── Xây dựng danh sách node cho mỗi cluster ───
    std::vector<std::vector<int>> sol(K);                // sol[k] = list node thuộc cluster k
    std::vector<char> included(N, 0);                    // đánh dấu node đã được gán chưa

    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];                               // cluster của node i

        if (c < 0 || c >= K)                             // cluster id ngoài phạm vi [0, K-1]
            return false;

        if (included[i])                                 // node bị gán 2 lần (defensive, không nên xảy ra)
            return false;

        included[i] = 1;                                 // đánh dấu đã gán
        sol[c].push_back(i);                             // thêm node i vào cluster c
    }

    // ─── Kiểm tra mọi node đều được gán ───
    if (std::any_of(included.begin(), included.end(),
                    [](char v) { return v == 0; }))      // có node nào chưa gán?
        return false;

    // ─── Kiểm tra ràng buộc trọng số cho mỗi cluster, mỗi chiều ───
    for (int k = 0; k < K; ++k)                          // duyệt mỗi cluster
    {
        for (int t = 0; t < M_weights; ++t)               // duyệt mỗi chiều trọng số
        {
            // Tính tổng trọng số chiều t của cluster k
            double wkt = 0.0;
            for (int v : sol[k])                          // duyệt mỗi node trong cluster k
                wkt += Wmat[v][t];                        // cộng trọng số

            // Tính relative tolerance
            // scale = max giữa 1.0, |wkt|, |lower|, |upper|
            // → tolerance tỷ lệ với giá trị lớn nhất → tránh false positive khi số lớn
            double scale = std::max(1.0,
                           std::max(std::abs(wkt),
                           std::max(std::abs(WLmat[k][t]),
                                    std::abs(WUmat[k][t]))));
            double tol = VALID_EPS * scale;               // tolerance thực tế

            // Kiểm tra: wkt + tol >= lower VÀ wkt - tol <= upper
            if (wkt + tol < WLmat[k][t] ||                // vi phạm lower bound
                wkt - tol > WUmat[k][t])                  // vi phạm upper bound
            {
                return false;                             // infeasible
            }
        }
    }

    return true;                                         // tất cả ràng buộc thỏa mãn → feasible
}


// ═══════════════════════════════════════════════════════════════════════════
// KHOẢNG CÁCH HAMMING GIỮA 2 NGHIỆM
//
// Đếm số node có cluster khác nhau giữa 2 nghiệm.
// Dùng để đo "khoảng cách" giữa các nghiệm trong population.
//
// Ví dụ: a = [0,1,2,0], b = [0,1,1,0] → khác ở index 2 → Hamming = 1
//
// Ứng dụng trong ACO:
//   - Chọn ant đa dạng để repair (tránh repair nhiều ant giống nhau)
//   - Xác định best local khác best global
// ═══════════════════════════════════════════════════════════════════════════
int hamming_distance(const vector<int>& a, const vector<int>& b)
{
    int diff = 0;                                        // đếm số vị trí khác nhau
    for (int i = 0; i < a.size(); ++i)                   // duyệt từng node
        if (a[i] != b[i])                                // nếu cluster khác nhau
            diff++;                                      // tăng đếm
    return diff;                                         // trả về khoảng cách Hamming
}


// ═══════════════════════════════════════════════════════════════════════════
// LƯU LOG RA FILE
//
// Ghi 3 file:
//   1. evolution: diễn biến qua các iteration (iter, time, cost, feasible,...)
//   2. objectives: cost tốt nhất cuối cùng
//   3. solutions: nghiệm tốt nhất (danh sách node mỗi cluster, 1-based)
// ═══════════════════════════════════════════════════════════════════════════
void SaveLogs(const ACOSolution &best)
{
    // ─── File 1: Evolution (diễn biến hội tụ) ───
    std::ofstream foutEvo(LOG_EVOL_FILENAME);            // mở file ghi evolution
    if (!foutEvo.is_open())                              // nếu không mở được
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_EVOL_FILENAME << " for writing.\n";
    }
    else
    {
        // Ghi header (dòng tiêu đề)
        foutEvo << "# iter   time(s)    bestCost    bestFeasible  bestThisIter  feasibleAnts  noImprove\n";

        // Ghi từng snapshot (mỗi 10 iteration được lưu vào log_rows)
        for (auto &r : log_rows)
        {
            foutEvo << std::setw(6) << r.iter                          // số iteration
                    << std::setw(12) << std::fixed << std::setprecision(4) << r.time       // thời gian (s)
                    << std::setw(14) << std::fixed << std::setprecision(6) << r.bestCost   // cost tốt nhất toàn cục
                    << std::setw(12) << (r.bestFeasible ? "1" : "0")                       // đã feasible chưa?
                    << std::setw(14) << std::fixed << std::setprecision(6) << r.bestThisIter // cost tốt nhất iteration này
                    << std::setw(12) << r.feasibleAnts                                      // số ant feasible
                    << std::setw(12) << r.noImprove                                         // số iteration liên tiếp không improve
                    << "\n";
        }
        foutEvo.close();                                 // đóng file
        std::cerr << "[SAVELOGS] evolution saved to " << LOG_EVOL_FILENAME << "\n";
    }

    // ─── File 2: Best cost (1 số duy nhất) ───
    std::ofstream foutCost(LOG_COST_FILENAME);
    if (!foutCost.is_open())
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_COST_FILENAME << " for writing.\n";
    }
    else
    {
        foutCost << std::fixed << std::setprecision(6) << best.cost << "\n";
                                                         // ghi cost tốt nhất
        foutCost.close();
        std::cerr << "[SAVELOGS] best cost saved to " << LOG_COST_FILENAME << "\n";
    }

    // ─── File 3: Best solution (danh sách cluster) ───
    std::ofstream foutSolu(LOG_SOLU_FILENAME);
    if (!foutSolu.is_open())
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_SOLU_FILENAME << " for writing.\n";
    }
    else
    {
        // Chuyển assign → danh sách node mỗi cluster (1-based cho dễ đọc)
        std::vector<std::vector<int>> clusters(K);       // clusters[k] = list node (1-based)
        for (int i = 0; i < N; ++i)
        {
            int c = (i < (int)best.assign.size()) ? best.assign[i] : -1;
                                                         // lấy cluster của node i (defensive)
            if (c >= 0 && c < K)
                clusters[c].push_back(i + 1);            // node 1-based (i+1)
        }

        // Ghi mỗi cluster 1 dòng, node sorted tăng dần
        for (int k = 0; k < K; ++k)
        {
            std::sort(clusters[k].begin(), clusters[k].end());
                                                         // sort node tăng dần
            for (int v : clusters[k])
                foutSolu << v << " ";                    // ghi node cách nhau bằng space
            foutSolu << "\n";                            // xuống dòng = cluster mới
        }
        foutSolu.close();
        std::cerr << "[SAVELOGS] best solution saved to " << LOG_SOLU_FILENAME << "\n";
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
// ══                                                                     ══
// ══   HÀM CHÍNH: ACO_tuned()                                          ══
// ══   Ant Colony Optimization cho bài toán MCGP                        ══
// ══                                                                     ══
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//
// FLOW TỔNG QUAN:
//   1. Khởi tạo dữ liệu toàn cục từ instance
//   2. Tính PENALTY_SCALE tự động theo instance
//   3. Khởi tạo ma trận pheromone phi[i][k]
//   4. Vòng lặp chính:
//      a. m con kiến xây nghiệm (construction phase)
//      b. Chọn top ants đa dạng → local search + tabu search (improvement phase)
//      c. Cập nhật best global
//      d. Cập nhật pheromone (evaporation + deposit)
//      e. Kiểm tra stagnation → reset nếu cần
//   5. Ghi log và trả kết quả
//
// THAM SỐ ĐẦU VÀO:
//   instance:         dữ liệu bài toán (N, K, M_weights, W, WL, WU, D)
//   maxIter:          số iteration tối đa
//   timeLimitSeconds: giới hạn thời gian (giây)
//   instance_name:    tên instance (dùng cho log filename)
//
// TRẢ VỀ:
//   ACOSolution: nghiệm tốt nhất tìm được (assign, cost, feasible, members,...)
// ═══════════════════════════════════════════════════════════════════════════

ACOSolution ACO_tuned(const Instance &instance, int maxIter, double timeLimitSeconds, const string &instance_name)
{
    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 0: CẤU HÌNH ĐƯỜNG DẪN LOG
    // ─────────────────────────────────────────────────────────────────────

    std::string base = parameters.LOGdir;                // lấy thư mục log từ config
    if (base.empty())
        base = "results/logs/aco_logs";                  // mặc định nếu không set
    if (base.back() == '/')
        base.pop_back();                                 // bỏ dấu / cuối nếu có

    // Tạo đường dẫn đầy đủ cho 3 file log
    LOG_EVOL_FILENAME = base + "/evolution/" + instance_name;    // file diễn biến hội tụ
    LOG_COST_FILENAME = base + "/objectives/" + instance_name;   // file cost tốt nhất
    LOG_SOLU_FILENAME = base + "/solutions/" + instance_name;    // file nghiệm tốt nhất


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 1: KIỂM TRA TÍNH HỢP LỆ CỦA INSTANCE
    //
    // Nếu tổng trọng số node không nằm trong [tổng lower, tổng upper]
    // → không thể tìm nghiệm feasible → dừng sớm
    // ─────────────────────────────────────────────────────────────────────

    if (!check_weights_validity(instance))
    {
        cerr << "[ERROR] Instance weight bounds inconsistent. Aborting ACO.\n";
        ACOSolution empty;                               // tạo nghiệm rỗng
        empty.assign.clear();                            // không có assignment
        empty.cost = 1e300;                              // cost "vô cực"
        empty.feasible = false;                          // infeasible
        return empty;                                    // trả về ngay
    }


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 2: COPY DỮ LIỆU INSTANCE VÀO BIẾN TOÀN CỤC
    //
    // Lý do dùng biến toàn cục: các module khác (local_search, tabu_search,
    // compute_cost, is_feasible) cần truy cập dữ liệu mà không cần
    // truyền qua tham số mỗi lần gọi.
    // ─────────────────────────────────────────────────────────────────────

    N = instance.nV;                                     // số node (vertices)
    K = instance.nK;                                     // số cluster (partitions)
    M_weights = instance.nT;                             // số chiều trọng số (resource types)

    // ═════ COPY TRỌNG SỐ NODE ═════
    // Wmat[i][t] = trọng số chiều t của node i
    Wmat.assign(N, vector<double>(M_weights, 0.0));      // khởi tạo N x M_weights, giá trị 0
    for (int i = 0; i < N; ++i)                          // duyệt mỗi node
        for (int t = 0; t < M_weights; ++t)               // duyệt mỗi chiều
            Wmat[i][t] = instance.W[i][t];                // copy từ instance

    // ═════ COPY LOWER/UPPER BOUND CLUSTER ═════
    // WLmat[k][t] = giới hạn dưới cluster k chiều t
    // WUmat[k][t] = giới hạn trên cluster k chiều t
    WLmat.assign(K, vector<double>(M_weights, 0.0));     // K x M_weights
    WUmat.assign(K, vector<double>(M_weights, 0.0));     // K x M_weights

    for (int k = 0; k < K; ++k)                          // duyệt mỗi cluster
        for (int t = 0; t < M_weights; ++t)               // duyệt mỗi chiều
        {
            WLmat[k][t] = instance.WL[k][t];             // lower bound
            WUmat[k][t] = instance.WU[k][t];             // upper bound
        }

    // ═════ COPY MA TRẬN KHOẢNG CÁCH ═════
    // distmat[i][j] = khoảng cách từ node i đến node j
    distmat.assign(N, vector<double>(N, 0.0));           // N x N
    for (int i = 0; i < N; ++i)                          // duyệt node nguồn
        for (int j = 0; j < N; ++j)                      // duyệt node đích
            distmat[i][j] = instance.D[i][j];            // copy

    // ═════ KIỂM TRA LẦN CUỐI ═════
    if (K <= 0 || N <= 0 || M_weights <= 0)              // dữ liệu không hợp lệ
    {
        cerr << "[ERROR] invalid N, K, or number of weight dimensions\n";
        ACOSolution empty;
        empty.assign.clear();
        empty.cost = 1e300;
        empty.feasible = false;
        return empty;
    }


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 3: TÍNH CÁC HẰNG SỐ SCALE TỰ ĐỘNG THEO INSTANCE
    //
    // PENALTY_SCALE và DIST_SCALE phụ thuộc vào đặc tính instance
    // để cân bằng giữa distance và penalty trong quá trình tìm kiếm.
    // ─────────────────────────────────────────────────────────────────────

    // ── 3a. Tính khoảng cách trung bình giữa các cặp node ──
    double BASE_COST = 0.0;                              // tổng khoảng cách tất cả cặp
    int pairCount = 0;                                   // số cặp

    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
        {
            BASE_COST += distmat[i][j];                  // cộng dist(i,j)
            pairCount++;                                 // đếm cặp
        }

    double meanDist = BASE_COST / pairCount;             // khoảng cách trung bình 1 cặp

    // ── 3b. Tính DIST_SCALE (heuristic normalization cho ACO construction) ──
    //
    // Khi xây nghiệm, mỗi node có sumDist[i][k] = tổng dist(i, mọi node trong cluster k)
    // Giá trị kỳ vọng ≈ avgClusterSize * meanDist
    // DIST_SCALE dùng để normalize giá trị này về khoảng [0, vài đơn vị]
    // → desirability heuristic không bị lệch scale

    double avgClusterSize = max(1.0, (double)N / K);     // kích thước cluster trung bình
    double DIST_SCALE = avgClusterSize * meanDist;       // scale factor cho distance heuristic

    // ── 3c. Tính PENALTY_SCALE (cân bằng penalty vs distance) ──
    //
    // Ý tưởng: penalty cho 1 đơn vị vi phạm trọng số nên "tương đương"
    //          với chi phí distance đáng kể, để ACO ưu tiên feasibility.
    //
    // PENALTY_SCALE = 50 * (meanDist / meanWeight)
    //   - meanDist lớn → penalty cần lớn theo
    //   - meanWeight nhỏ → 1 đơn vị vi phạm "đắt" hơn → penalty scale tăng

    double sumW = 0.0;                                   // tổng tất cả trọng số
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            sumW += Wmat[i][t];                          // cộng dồn

    double meanWeight = sumW / (N * M_weights);          // trọng số trung bình 1 node 1 chiều

    PENALTY_SCALE = 50.0 * (meanDist / meanWeight);      // auto-scale penalty
    // Ví dụ: meanDist=100, meanWeight=10 → PENALTY_SCALE=500
    //         → 1 đơn vị vi phạm trọng số bị phạt 500 đơn vị distance


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 4: CẤU HÌNH THAM SỐ ACO
    //
    // Các tham số điều khiển hành vi thuật toán kiến
    // ─────────────────────────────────────────────────────────────────────

    int m = min(N / 2, 40);                              // số con kiến (ants) mỗi iteration
                                                         // min(N/2, 40): scale theo bài toán nhưng cap ở 40
                                                         // Nhiều ant → explore tốt hơn nhưng chậm hơn

    double alpha = 1.25;                                 // hệ số importance của PHEROMONE
                                                         // alpha lớn → kiến ưu tiên đường có nhiều pheromone
                                                         // (exploitation > exploration)

    double beta = 1.0;                                   // hệ số importance của HEURISTIC (desirability)
                                                         // beta lớn → kiến ưu tiên cluster "tốt" về distance + capacity
                                                         // (greedy hơn)

    double rho = 0.2;                                    // tỷ lệ bay hơi pheromone (evaporation rate)
                                                         // Mỗi iteration: phi *= (1-rho)
                                                         // rho lớn → quên nhanh → explore nhiều
                                                         // rho nhỏ → nhớ lâu → exploit nhiều

    // ── Tham số adaptive (thay đổi theo quá trình chạy) ──

    double T_max = 1.0;                                  // lượng pheromone deposit tối đa (best ant)
    double T_min = 0.1;                                 // lượng pheromone deposit tối thiểu (all ants)
                                                         // T_max >> T_min → best ant ảnh hưởng mạnh
    const double PHI_MIN = 0.05;                         // giới hạn dưới vết mùi (tránh = 0)
    const double PHI_MAX = 5;                            // giới hạn trên vết mùi (tránh quá lớn → bias cực)

    double Q_max = 0.8;                                  // xác suất exploitation tối đa
    double Q_min = 0.1;                                  // xác suất exploitation tối thiểu
    double Q0 = Q_max;                                   // xác suất exploitation hiện tại
                                                         // Q0 cao → kiến thường chọn cluster tốt nhất (exploit)
                                                         // Q0 thấp → kiến chọn theo roulette wheel (explore)

    double STAGNATE_DROP = 0.05;                         // Ý định: mỗi iteration stagnate, giảm Q0 đi 0.05

    int STAGNATE_LIMIT = 10;                             // sau bao nhiêu iteration không improve → bắt đầu giảm Q0
                                                         // → chuyển dần từ exploitation sang exploration

    // ── Repair configuration ──
    int repairTop = 5;                                   // số ant được chọn để local search + tabu search
                                                         // Không repair TẤT CẢ m ant (quá chậm)
                                                         // Chỉ repair top ants (theo cost) + đa dạng (Hamming)


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 5: KHỞI TẠO RNG VÀ BIẾN TRẠNG THÁI
    // ─────────────────────────────────────────────────────────────────────

    mt19937_64 rng(                                      // Mersenne Twister 64-bit random generator
        (unsigned)chrono::high_resolution_clock::now()
            .time_since_epoch().count());                 // seed = thời gian hiện tại (nanoseconds)
                                                         // → mỗi lần chạy cho kết quả khác nhau

    uniform_real_distribution<double> uni01(0.0, 1.0);   // phân phối đều trong [0, 1)
                                                         // dùng cho: exploitation/exploration decision,
                                                         //           roulette wheel selection

    ACOSolution best;                                    // nghiệm tốt nhất tìm được (global best)
                                                         // Khởi tạo mặc định: cost = 1e300, feasible = false
                                                         // (sẽ được cập nhật ngay iteration đầu tiên)

    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 6: KHỞI TẠO MA TRẬN PHEROMONE
    //
    // phi[i][k] = lượng pheromone trên cạnh "gán node i vào cluster k"
    //
    // Giá trị cao → kiến có xu hướng chọn cluster k cho node i
    // Ban đầu: tất cả bằng T_min (uniform, không bias)
    // Sau mỗi iteration: evaporate + deposit dựa trên best solution
    // ─────────────────────────────────────────────────────────────────────

    vector<vector<double>> phi(N, vector<double>(K, T_min));
                                                         // phi[i][k] khởi tạo = T_min cho tất cả (i,k)
                                                         // Kích thước: N x K

    auto start = Clock::now();                           // thời điểm bắt đầu (để đo elapsed time)
    int iter = 0;                                        // đếm iteration
    int noImprove = 0;                                   // số iteration liên tiếp không cải thiện best


    // ═══════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════
    // ══                                                                   ══
    // ══   VÒNG LẶP CHÍNH ACO                                             ══
    // ══                                                                   ══
    // ══   Mỗi iteration:                                                  ══
    // ══     Phase 1: Construction — m kiến xây nghiệm                    ══
    // ══     Phase 2: Improvement — top ants được local search + tabu     ══
    // ══     Phase 3: Update — cập nhật best global + pheromone           ══
    // ══                                                                   ══
    // ═══════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════

    while (iter < maxIter &&                             // chưa hết iteration
           chrono::duration<double>(Clock::now() - start).count() < timeLimitSeconds)
                                                         // chưa hết thời gian
    {
        ++iter;                                          // tăng đếm iteration

        // ═════════════════════════════════════════════════════════════════
        // PHASE 1: CONSTRUCTION — Mỗi kiến xây 1 nghiệm hoàn chỉnh
        //
        // Mỗi kiến duyệt N node (theo thứ tự ngẫu nhiên).
        // Với mỗi node i, chọn cluster k dựa trên:
        //   probability ∝ tau[i][k]^alpha * desirability[i][k]^beta
        //
        //   tau[i][k] = phi[i][k]     (pheromone, học từ best solutions)
        //   desirability = f(distance, capacity fit, violation)
        //
        // Quyết định exploit (chọn k tốt nhất) hay explore (roulette wheel)
        // dựa trên Q0 (exploitation probability).
        // ═════════════════════════════════════════════════════════════════

        vector<ACOSolution> ants(m);                     // mảng m con kiến

        for (int a = 0; a < m; ++a)                      // xây nghiệm cho từng kiến
        {
            // ── Khởi tạo nghiệm rỗng cho kiến a ──

            ants[a].assign.assign(N, -1);                // assign[i] = -1 (chưa gán cluster nào)
                                                         // Kích thước N

            ants[a].members.assign(K, vector<int>());    // members[k] = {} (mỗi cluster rỗng)
                                                         // Kích thước K

            ants[a].clusterWeight.assign(K, vector<double>(M_weights, 0.0));
                                                         // clusterWeight[k][t] = 0 (tổng trọng số = 0)
                                                         // Kích thước K x M_weights

            ants[a].clusterSumDist.assign(N, vector<double>(K, 0.0));
                                                         // clusterSumDist[i][k] = 0
                                                         // = tổng dist(i, mọi node đã ở cluster k)
                                                         // Ban đầu = 0 vì chưa node nào được gán
                                                         // Kích thước N x K

            // ── Tạo thứ tự node ngẫu nhiên ──
            // Mỗi kiến duyệt node theo thứ tự khác nhau
            // → đa dạng hóa nghiệm (node gán trước ảnh hưởng node gán sau)
            vector<int> nodes(N);                        // [0, 1, 2, ..., N-1]
            iota(nodes.begin(), nodes.end(), 0);         // fill giá trị 0..N-1
            shuffle(nodes.begin(), nodes.end(), rng);    // xáo trộn ngẫu nhiên

            // ── Duyệt từng node theo thứ tự ngẫu nhiên và gán vào cluster ──
            for (int idx = 0; idx < N; ++idx)            // idx = thứ tự xây dựng (0..N-1)
            {
                int i = nodes[idx];                      // node thực sự đang xét

                double bestWeight = -1.0;                // trọng số max (cho exploitation)
                int chosenK = 0;                         // cluster sẽ được chọn
                vector<double> weights(K, 0.0);          // weights[k] = probability weight cho cluster k

                // ── Tính desirability + probability weight cho mỗi cluster k ──
                for (int k = 0; k < K; ++k)
                {
                    // ~~~~ HEURISTIC 1: CAPACITY FIT ~~~~
                    // Đánh giá node i "hợp" với cluster k như thế nào về mặt trọng số

                    double dot = 0.0;                    // tích vô hướng (dot product) giữa need và weight
                    double normNeed = 0.0;               // norm² của vector need
                    double normNode = 0.0;               // norm² của vector weight node
                    double emptiness = 0.0;              // mức "đói" tương đối của cluster
                    double overflow = 0.0;               // mức vi phạm upper bound

                    for (int t = 0; t < M_weights; ++t)  // duyệt mỗi chiều trọng số
                    {
                        // need = lượng trọng số cluster k CÒN THIẾU để đạt lower bound
                        // Nếu cluster đã đủ/thừa → need = 0
                        double need = max(0.0, WLmat[k][t] - ants[a].clusterWeight[k][t]);
                                                         // max(0, lower - current)

                        double w = Wmat[i][t];           // trọng số node i tại chiều t

                        // Kiểm tra nếu gán node i vào cluster k có vượt upper bound không
                        double after = ants[a].clusterWeight[k][t] + w;
                                                         // trọng số cluster k SAU khi thêm node i

                        if (after > WUmat[k][t])         // vượt upper bound
                        {
                            double excess = (after - WUmat[k][t]) / WUmat[k][t];
                                                         // tỷ lệ vượt quá (normalized)
                            overflow += excess;          // cộng dồn tỷ lệ vượt
                        }

                        // ── Vector Fit: cosine similarity giữa "need" và "weight" ──
                        //
                        // Ý tưởng: nếu node i có profile trọng số GIỐNG với phần
                        //          cluster k đang thiếu → fit tốt
                        //
                        // Ví dụ: cluster cần [100, 0], node có [80, 5]
                        //        → cosine cao (hướng tương tự)
                        //        cluster cần [100, 0], node có [5, 80]
                        //        → cosine thấp (hướng khác)
                        dot      += need * w;            // tử số cosine similarity
                        normNeed += need * need;         // mẫu số (phần need)
                        normNode += w * w;               // mẫu số (phần node weight)

                        // ── Emptiness: mức "đói" tương đối của cluster ──
                        // emptiness += need / WLmat[k][t]
                        // = tỷ lệ phần trăm còn thiếu so với lower bound
                        // Cluster đang rất đói → emptiness cao → ưu tiên gán node vào
                        emptiness += need / max(WLmat[k][t], VALID_EPS); // ⚠️ Potential div by 0 nếu WLmat=0
                    }

                    // ── Normalize ──
                    double vectorFit = 0.0;              // cosine similarity [0, 1]
                    if (normNeed > 1e-12 && normNode > 1e-12)
                        vectorFit = dot / (sqrt(normNeed) * sqrt(normNode));
                                                         // cos(θ) = (need·w) / (||need|| * ||w||)
                                                         // = 1 nếu cùng hướng hoàn toàn
                                                         // = 0 nếu vuông góc

                    emptiness /= M_weights;              // trung bình emptiness trên các chiều [0, 1]
                    overflow /= M_weights;               // trung bình overflow trên các chiều [0, ~∞)

                    // ── Kết hợp thông tin capacity ──
                    double capacityGain =
                        0.8 * vectorFit +                // 80% weight: node i fit tốt với nhu cầu cluster k
                        0.3 * emptiness;                 // 30% weight: cluster k đang đói
                                                         // Tổng có thể > 1 (không phải xác suất, là score)

                    // ── Penalty cho overflow ──
                    // Nếu gán node i vào k làm vượt upper → giảm desirability
                    // exp(-4*overflow): overflow=0 → 1.0, overflow=1 → 0.018
                    double violationPenalty = exp(-4.0 * overflow);


                    // ~~~~ HEURISTIC 2: DISTANCE ~~~~
                    // Node i nên gán vào cluster k có tổng khoảng cách nhỏ

                    // distTerm = sumDist[i][k] / DIST_SCALE
                    //   sumDist[i][k] = tổng dist(i, mọi node đã ở cluster k)
                    //   DIST_SCALE normalize về khoảng ~1
                    double distTerm = ants[a].clusterSumDist[i][k] / DIST_SCALE;


                    // ~~~~ KẾT HỢP HEURISTIC ~~~~
                    //
                    // desirability = (1 / (1 + distTerm))     ← distance nhỏ → desir cao
                    //              * (1 + capacityGain)       ← capacity fit tốt → desir cao
                    //              * violationPenalty          ← overflow → desir giảm
                    double desir =
                        (1.0 / (1.0 + distTerm)) *       // distance term: nhỏ → gần → tốt
                        (1.0 + capacityGain) *            // capacity term: fit → tốt
                        violationPenalty;                  // violation term: overflow → xấu


                    // ~~~~ PHEROMONE ~~~~
                    double tau = phi[i][k];               // pheromone trên cạnh (node i, cluster k)
                                                         // Cao → best solutions thường gán i vào k


                    // ~~~~ PROBABILITY WEIGHT ~~~~
                    // Theo công thức ACO chuẩn:
                    //   weight[k] = tau^alpha * desirability^beta
                    //
                    // alpha = 1.25: pheromone quan trọng hơn heuristic một chút
                    // beta = 1.0:   heuristic importance bình thường
                    double weight = pow(tau, alpha) * pow(desir, beta);
                    weights[k] = weight;                  // lưu weight cho roulette wheel

                    // Track cluster có weight cao nhất (cho exploitation)
                    if (weight > bestWeight)
                    {
                        bestWeight = weight;
                        chosenK = k;                      // cluster tốt nhất hiện tại
                    }
                }
                // ── Kết thúc tính weight cho K cluster ──


                // ═════════════════════════════════════════════════════════
                // QUY TẮC CHỌN CLUSTER: Exploitation vs Exploration
                //
                // Với xác suất Q0: EXPLOITATION
                //   → chọn cluster có weight cao nhất (chosenK đã set ở trên)
                //   → greedy, exploit kinh nghiệm từ pheromone
                //
                // Với xác suất (1-Q0): EXPLORATION
                //   → chọn theo Roulette Wheel Selection
                //   → probability(k) ∝ weights[k]
                //   → có cơ hội thử cluster mới (diversity)
                //
                // Ban đầu Q0 = Q_max = 0.8 → 80% exploit, 20% explore
                // Khi stagnate → Q0 giảm → explore nhiều hơn
                // ═════════════════════════════════════════════════════════

                double q = uni01(rng);                    // random trong [0, 1)

                if (q >= Q0)                              // q >= Q0 → EXPLORATION (roulette wheel)
                                                         // ⚠️ Logic: q >= Q0 → explore
                                                         //   Q0 = 0.8 → 20% explore (q ∈ [0.8, 1))
                                                         //   → đúng: phần lớn exploit, ít explore
                {
                    double sumW = accumulate(weights.begin(), weights.end(), 0.0);
                                                         // tổng weight tất cả cluster
                    if (sumW > 0)                        // tránh chia 0
                    {
                        double pick = uni01(rng) * sumW;  // random point trên wheel
                        double acc = 0.0;                 // tích lũy
                        for (int k = 0; k < K; ++k)
                        {
                            acc += weights[k];            // cộng dồn weight
                            if (pick <= acc)              // nếu vượt qua → chọn cluster k
                            {
                                chosenK = k;
                                break;
                            }
                        }
                    }
                    // Nếu sumW == 0 → giữ chosenK từ argmax (fallback)
                }
                // Nếu q < Q0 → EXPLOITATION → giữ chosenK = argmax(weights)

                // ═════════════════════════════════════════════════════════
                // GÁN NODE i VÀO CLUSTER chosenK
                // Cập nhật tất cả cấu trúc dữ liệu của ant
                // ═════════════════════════════════════════════════════════

                ants[a].assign[i] = chosenK;             // ghi nhận: node i thuộc cluster chosenK

                ants[a].members[chosenK].push_back(i);   // thêm node i vào danh sách member của cluster

                // Cập nhật tổng trọng số cluster
                for (int t = 0; t < M_weights; ++t)
                    ants[a].clusterWeight[chosenK][t] += Wmat[i][t];
                                                         // clusterWeight[k][t] += weight node i chiều t

                // Cập nhật sumDist cho TẤT CẢ node j
                // Vì node i vừa được thêm vào cluster chosenK:
                //   sumDist[j][chosenK] += dist(j, i) cho mọi j
                // → bất kỳ node j nào, tổng dist tới cluster chosenK tăng thêm dist(j,i)
                for (int j = 0; j < N; ++j)              // duyệt mọi node
                    ants[a].clusterSumDist[j][chosenK] += distmat[j][i];

            }
            // ── Kết thúc gán N node cho ant a ──

            // ═════ TÍNH COST CHO ANT SAU KHI XÂY XONG ═════
            // cost = intra_distance + PENALTY_SCALE * total_violation
            ants[a].cost = compute_cost(ants[a].assign);
            ants[a].feasible = is_feasible(ants[a].assign);

        } // ── Kết thúc construction phase (m ants built) ──


        // ═════════════════════════════════════════════════════════════════
        // PHASE 2: IMPROVEMENT — Local Search + Tabu Search cho top ants
        //
        // Không improve TẤT CẢ m ants (quá chậm).
        // Chọn repairTop ants TỐT NHẤT + ĐA DẠNG:
        //   - Sort ants theo cost tăng dần
        //   - Chọn ant tốt nhất, nếu ant tiếp theo đủ khác biệt
        //     (Hamming distance >= MIN_DIFF) thì thêm vào
        //   - Đảm bảo đa dạng để tránh converge sớm
        // ═════════════════════════════════════════════════════════════════

        // ── Sort ants theo cost ──
        vector<int> order(m);                            // index [0, 1, ..., m-1]
        iota(order.begin(), order.end(), 0);             // fill
        sort(order.begin(), order.end(), [&](int a1, int a2)
             { return ants[a1].cost < ants[a2].cost; }); // sort tăng dần theo cost
                                                         // order[0] = ant có cost thấp nhất

        // ── Chọn repairTop ants đa dạng ──
        vector<int> selected;                            // index các ant được chọn để repair
        int MIN_DIFF = max(N/100,1);                                // Hamming distance tối thiểu giữa các ant selected

        for (int idx = 0; idx < m && selected.size() < repairTop; ++idx)
        {
            int ai = order[idx];                         // ant tốt thứ idx
            bool diverse = true;                         // giả sử đủ khác biệt

            // Kiểm tra Hamming distance với tất cả ant đã chọn
            for (int sj : selected)
            {
                if (hamming_distance(ants[ai].assign, ants[sj].assign) < MIN_DIFF)
                {
                    diverse = false;                     // quá giống ant đã chọn → bỏ qua
                    break;
                }
            }

            if (diverse)
                selected.push_back(ai);                  // thêm vào danh sách repair
        }

        // Nếu chưa đủ repairTop ant (do tất cả quá giống nhau)
        // → bổ sung thêm ant theo thứ tự cost (không check Hamming)
        for (int idx = 0; idx < m && selected.size() < repairTop; ++idx)
        {
            int ai = order[idx];
            if (find(selected.begin(), selected.end(), ai) == selected.end())
                                                         // nếu ai chưa trong selected
                selected.push_back(ai);                  // thêm vào
        }

        // ── Chạy Local Search + Tabu Search cho các ant được chọn ──
        for (int ai : selected)
        {
            // Local Search: relocate + swap (tối đa 1000 moves)
            // Cải thiện nghiệm bằng các thay đổi nhỏ (neighborhood search)
            local_search(ants[ai], rng, 1000);

            // Iterated Tabu Search: tìm kiếm sâu hơn với tabu list
            // (tránh quay lại nghiệm đã thăm gần đây)
            iterated_tabu_search(ants[ai], rng);

            // Tính lại cost và feasibility sau khi improve
            ants[ai].cost = compute_cost(ants[ai].assign);
            ants[ai].feasible = is_feasible(ants[ai].assign);
        }


        // ═════════════════════════════════════════════════════════════════
        // PHASE 3: UPDATE — Cập nhật best global + pheromone
        // ═════════════════════════════════════════════════════════════════

        // ── Re-sort ants sau local search (cost có thể thay đổi) ──
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a1, int a2)
             { return ants[a1].cost < ants[a2].cost; });

        // ── Cập nhật global best ──
        //
        // Logic ưu tiên:
        //   1. Feasible LUÔN thắng infeasible (dù cost cao hơn)
        //   2. Trong cùng loại (cả 2 feasible hoặc cả 2 infeasible): cost nhỏ hơn thắng
        bool improvedThisIter = false;

        for (int r = 0; r < m; ++r)                      // duyệt tất cả ant (theo thứ tự cost)
        {
            int ai = order[r];

            bool curFeasible  = ants[ai].feasible;       // ant hiện tại có feasible?
            double curCost    = ants[ai].cost;            // cost ant hiện tại

            bool bestFeasible = best.feasible;            // best global có feasible?
            double bestCost   = best.cost;                // cost best global

            bool accept = false;                         // có chấp nhận ant này làm best mới?

            if (curFeasible)
            {
                // Trường hợp 1: Ant mới FEASIBLE
                //   → Chấp nhận nếu:
                //     (a) Best cũ infeasible (feasible LUÔN thắng infeasible)
                //     (b) Hoặc cost mới < cost cũ
                if (!bestFeasible || curCost + 1e-12 < bestCost)
                    accept = true;
                // ⚠️ 1e-12 tolerance: tránh chấp nhận do lỗi floating point
            }
            else
            {
                // Trường hợp 2: Ant mới INFEASIBLE
                //   → Chỉ chấp nhận nếu best cũng infeasible VÀ cost nhỏ hơn
                //   → KHÔNG BAO GIỜ thay thế best feasible bằng infeasible
                if (!bestFeasible && curCost + 1e-12 < bestCost)
                    accept = true;
            }

            if (accept)
            {
                best = ants[ai];                         // cập nhật best global
                improvedThisIter = true;                 // đánh dấu iteration này có cải thiện

                Q0 = Q_max;                              // reset Q0 về max (exploit nghiệm tốt)
                                                         // → sau khi tìm được best mới, tập trung exploit

                noImprove = 0;                           // reset counter stagnation

                // In thông báo ra stderr
                auto now = Clock::now();
                double elapsed = chrono::duration<double>(now - start).count();
                cerr << "[ITER " << iter << "] New best cost=" << format_cost_with_commas(best.cost, 0)
                     << " (feasible=" << (best.feasible ? "YES" : "NO")
                     << ", time " << elapsed << "s)\n";
            }
        }

        // Nếu iteration này KHÔNG cải thiện best
        if (!improvedThisIter)
        {
            ++noImprove;                                 // tăng counter stagnation

            // Nếu stagnate quá lâu → giảm Q0 để explore nhiều hơn
            if (noImprove > STAGNATE_LIMIT)
            {
                Q0 -= STAGNATE_DROP;                     // ⚠️ STAGNATE_DROP = 0 (do bug int)
                                                         //    → Q0 không thực sự giảm!
                                                         //    Fix: đổi STAGNATE_DROP thành double
                if (Q0 < Q_min)
                    Q0 = Q_min;                          // clamp: không giảm dưới Q_min
            }
        }


        // ═════════════════════════════════════════════════════════════════
        // PHEROMONE UPDATE
        //
        // Gồm 3 bước:
        //   1. Evaporation: phi[i][k] *= (1 - rho)
        //      → pheromone bay hơi, quên dần nghiệm cũ
        //
        //   2. Deposit từ BEST GLOBAL:
        //      phi[i][c] += T_max  (c = cluster của node i trong best solution)
        //      → tăng cường pheromone trên các cạnh (i, k) trong nghiệm tốt nhất
        //      → kiến tương lai có xu hướng đi theo best solution
        //
        //   3. Deposit nhỏ cho TẤT CẢ cạnh:
        //      phi[i][k] += T_min
        //      → đảm bảo không cạnh nào bị pheromone = 0
        //      → luôn có cơ hội nhỏ để explore mọi hướng
        //
        //   4. Clamp: PHI_MIN <= phi[i][k] <= PHI_MAX
        //      → tránh giá trị cực đoan (overflow/underflow)
        // ═════════════════════════════════════════════════════════════════

        // Bước 1: EVAPORATION — bay hơi pheromone
        for (int i = 0; i < N; ++i)                      // duyệt mỗi node
            for (int k = 0; k < K; ++k)                  // duyệt mỗi cluster
                phi[i][k] *= (1.0 - rho);                // phi *= 0.8 (giảm 20%)

        // Bước 2: DEPOSIT từ best global
        // Chỉ best global được deposit T_max (Max-Min Ant System style)
        for (int i = 0; i < N; ++i)                      // duyệt mỗi node
        {
            int c = best.assign[i];                      // cluster của node i trong best solution
            if (c >= 0 && c < K)                         // kiểm tra hợp lệ
                phi[i][c] += T_max;                      // deposit lượng lớn trên cạnh best
        }

        // ── Tìm Best Local (ant tốt nhất KHÁC best global) ──
        // Dùng cho logging, có thể dùng để deposit bổ sung
        int bestLocal = -1;

        for (int r = 0; r < m; ++r)
        {
            int ai = order[r];                           // ant thứ r (theo cost tăng dần)

            // Bỏ qua nếu quá giống best global (Hamming < MIN_DIFF)
            if (hamming_distance(ants[ai].assign, best.assign) < MIN_DIFF)
                continue;

            bestLocal = ai;                              // ant tốt nhất khác best global
            break;                                       // chỉ cần 1 cái
        }

        // Bước 4: CLAMP pheromone vào [PHI_MIN, PHI_MAX]
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                phi[i][k] = max(PHI_MIN, min(PHI_MAX, phi[i][k]));
                                                         // clamp: PHI_MIN <= phi <= PHI_MAX


        // ═════════════════════════════════════════════════════════════════
        // LOGGING (mỗi 10 iteration)
        // ═════════════════════════════════════════════════════════════════

        if (iter % 10 == 0)                              // log mỗi 10 iteration
        {
            // Best local ant (nếu không tìm được → dùng ant tốt nhất)
            int safeLocal = (bestLocal >= 0) ? bestLocal : order[0];
            auto bestThisIter = ants[safeLocal];

            // Đếm số ant feasible trong iteration này
            int feasCount = 0;
            for (int a = 0; a < m; ++a)
                if (ants[a].feasible)
                    feasCount++;

            // In ra stderr
            double elapsed = chrono::duration<double>(Clock::now() - start).count();
            cerr << "[ITER " << iter << "] bestGlobalCost=" << format_cost_with_commas(best.cost, 0)
                 << " (feasible=" << (best.feasible ? "YES" : "NO") << ")"
                 << " bestThisIter=" << format_cost_with_commas(bestThisIter.cost, 0)
                 << " feasibleAnts=" << feasCount
                 << " noImprove=" << noImprove
                 << " (elapsed " << elapsed << "s)\n";

            // Lưu snapshot vào log_rows (sẽ ghi ra file ở cuối)
            LogRow r;
            r.iter = iter;                               // iteration number
            r.time = elapsed;                            // elapsed time
            r.bestCost = best.cost;                      // global best cost
            r.bestFeasible = best.feasible;              // global best feasible?
            r.bestThisIter = bestThisIter.cost;          // best cost iteration này
            r.feasibleAnts = feasCount;                  // số ant feasible
            r.noImprove = noImprove;                     // iteration stagnation
            log_rows.push_back(r);                       // thêm vào danh sách log
        }


        // ═════════════════════════════════════════════════════════════════
        // STAGNATION RESET
        //
        // Nếu không cải thiện best sau noImproveReset iteration liên tiếp:
        //   → Reset pheromone về T_min (uniform, như ban đầu)
        //   → Reset Q0 về Q_max
        //   → Cho phép ACO "khám phá lại từ đầu" với kinh nghiệm mới
        //
        // Đây là cơ chế thoát local optima quan trọng nhất.
        // ═════════════════════════════════════════════════════════════════

        int noImproveReset = 200;                        // ngưỡng reset (200 iteration)

        if (noImprove >= noImproveReset)
        {
            cerr << "[RESET] no improvement for" << noImproveReset << "-> reset pheromones\n";

            // Reset toàn bộ pheromone về T_min
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < K; ++k)
                    phi[i][k] = T_min;                   // phi = 0.05 (uniform)

            noImprove = 0;                               // reset counter
            Q0 = Q_max;                                  // reset exploitation probability
        }

    } // ════════ KẾT THÚC VÒNG LẶP ACO ════════


    // ═══════════════════════════════════════════════════════════════════════
    // BƯỚC CUỐI: IN KẾT QUẢ VÀ GHI LOG
    // ═══════════════════════════════════════════════════════════════════════

    // In nghiệm tốt nhất ra stderr (danh sách node mỗi cluster, 1-based)
    vector<vector<int>> clusters(K);
    for (int i = 0; i < N; ++i)
    {
        int c = (i < (int)best.assign.size()) ? best.assign[i] : -1;
        if (c >= 0 && c < K)
            clusters[c].push_back(i + 1);                // node 1-based
    }
    for (int k = 0; k < K; ++k)
    {
        for (int node : clusters[k])
            cerr << node << " ";                         // in node thuộc cluster k
        cerr << "\n";                                    // xuống dòng = cluster mới
    }

    // In trạng thái feasibility và cost ra stdout
    if (!best.feasible)
        cout << "Final solution is invalid.\n";
    else
        cout << "Final solution is valid.\n";

    cout << "Final cost = " << format_cost_with_commas(best.cost, 0) << "\n";

    // Ghi tất cả log ra file
    SaveLogs(best);

    return best;                                         // trả nghiệm tốt nhất
}
