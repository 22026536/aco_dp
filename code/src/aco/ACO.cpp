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

vector<vector<int>> globalCL;   // candidate list toàn cục
const int GLOBAL_CL_SIZE = 20;  // só lượng candidate list cho mỗi đỉnh

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
// FAST INCREMENTAL COST — tính cost từ sumDist đã có sẵn
//
// Thay vì O(N²) mỗi lần gọi compute_cost, ta dùng sumDist đã maintain
// trong ACOSolution để tính trong O(N + K*M_weights)
//
// Tại sao nhanh hơn:
//   intra = Σ_i sumDist[i][assign[i]] / 2
//   (chia 2 vì mỗi cặp được đếm 2 lần: i→j trong sumDist[i] và j→i trong sumDist[j])
//
// Lưu ý: chỉ dùng được khi sumDist đã được maintain chính xác
// ═══════════════════════════════════════════════════════════════════════════
double compute_cost_fast(const ACOSolution &sol)
{
    // Tính intra-distance từ sumDist: O(N)
    double intra = 0.0;
    for (int i = 0; i < N; ++i) {
        int k = sol.assign[i];
        if (k < 0 || k >= K) return 1e300;
        intra += sol.clusterSumDist[i][k];
        // sumDist[i][k] = Σ dist(i,j) cho j ∈ cluster k
        // Nhưng dist(i,j) + dist(j,i) cần đếm cả 2 chiều
        // Và mỗi cặp bị đếm 2 lần (từ i và từ j)
    }
    // intra bây giờ = Σ_i Σ_{j∈same_cluster, j≠i} dist(i,j)
    //               = Σ_{i<j, same cluster} (dist(i,j) + dist(j,i))
    // → đúng rồi, KHÔNG cần chia 2 vì distmat[i][j] ≠ distmat[j][i] có thể

    // Nhưng chờ đã - sumDist[i][k] bao gồm dist(i,i)=0 nếu i ∈ cluster k
    // → OK vì dist(i,i) = 0

    // Thực ra cần kiểm tra lại: compute_cost gốc tính Σ_{i<j} (d[i][j]+d[j][i])
    // sumDist[i][k] = Σ_{j∈k} d[i][j], kể cả j=i (nhưng d[i][i]=0)
    // Σ_i sumDist[i][assign[i]] = Σ_i Σ_{j∈same_cluster} d[i][j]
    //   = Σ_{i,j cùng cluster, i≠j} d[i][j]
    //   = Σ_{i<j, cùng cluster} (d[i][j] + d[j][i])  ← chính xác!

    // Tính violation: O(K * M_weights)
    double total_violation = 0.0;
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < M_weights; ++t) {
            double s = sol.clusterWeight[k][t];
            if (s < WLmat[k][t]) total_violation += (WLmat[k][t] - s);
            if (s > WUmat[k][t]) total_violation += (s - WUmat[k][t]);
        }

    return intra + total_violation * PENALTY_SCALE;
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
    // BƯỚC 1: CẤU HÌNH ĐƯỜNG DẪN LOG
    // ─────────────────────────────────────────────────────────────────────

    std::string base = parameters.LOGdir;                // lấy thư mục log từ config
    if (base.empty())
        base = "results/logs";                  // mặc định nếu không set
    if (base.back() == '/')
        base.pop_back();                                 // bỏ dấu / cuối nếu có
 
    // Tạo đường dẫn đầy đủ cho 3 file log
    // Cấu trúc: base/instance_name/evolution/instance_name, v.v.
    std::string instDir = base + "/" + instance_name;
    LOG_EVOL_FILENAME = instDir + "/evolution/" + instance_name;  // file diễn biến hội tụ
    LOG_COST_FILENAME = instDir + "/objectives/" + instance_name; // file cost tốt nhất
    LOG_SOLU_FILENAME = instDir + "/solutions/" + instance_name;  // file nghiệm tốt nhất
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
            BASE_COST += distmat[i][j] + distmat[j][i];  // cộng dist(i,j)
            pairCount++;                                 // đếm cặp
        }

    double meanDist = BASE_COST / pairCount;             // khoảng cách trung bình 1 cặp

    // ── 3b. Tính DIST_SCALE (heuristic normalization cho ACO construction) ──
    //
    // Khi xây nghiệm, mỗi node có sumDist[i][k] = tổng dist(i, mọi node trong cluster k)
    // Giá trị kỳ vọng ≈ avgClusterSize * meanDist
    // DIST_SCALE dùng để normalize giá trị này về khoảng [0, vài đơn vị]
    // → desirability heuristic không bị lệch scale

    double avgClusterSize = N / K;     // kích thước cluster trung bình
    double DIST_SCALE = avgClusterSize * meanDist;       // scale factor cho distance heuristic

    // ── 3c. Tính PENALTY_SCALE (cân bằng penalty vs distance) ──
    //
    // Ý tưởng: penalty cho 1 đơn vị vi phạm trọng số nên "tương đương"
    //          với chi phí distance đáng kể, để ACO ưu tiên feasibility.
    //
    // PENALTY_SCALE = ((N / K) / M_weights) * (meanDist / meanWeight)
    //   - meanDist lớn → penalty cần lớn theo
    //   - meanWeight nhỏ → 1 đơn vị vi phạm "đắt" hơn → penalty scale tăng

    double sumAllW = 0.0;                                  // tổng tất cả trọng số
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            sumAllW += Wmat[i][t];                          // cộng dồn

    double meanWeight = max(sumAllW / (N * M_weights), 1e-12);          // trọng số trung bình 1 node 1 chiều

    //         → 1 đơn vị vi phạm trọng số bị phạt x đơn vị distance
    PENALTY_SCALE = ((N / K)) * (meanDist / meanWeight);

    // Build candidate list
    globalCL.assign(N, vector<int>(GLOBAL_CL_SIZE));
    {
        vector<pair<double,int>> tmp(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                tmp[j] = {(distmat[i][j]+distmat[j][i])*0.5, j};
            tmp[i].first = 1e300;
            partial_sort(tmp.begin(), tmp.begin()+GLOBAL_CL_SIZE, tmp.end());
            for (int r = 0; r < GLOBAL_CL_SIZE; ++r)
                globalCL[i][r] = tmp[r].second;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 4: CẤU HÌNH THAM SỐ ACO
    //
    // Các tham số điều khiển hành vi thuật toán kiến
    // ─────────────────────────────────────────────────────────────────────

    int m = 40;                                         // số con kiến (ants) mỗi iteration
                                                        // min(N/2, 40): scale theo bài toán nhưng cap ở 40
                                                        // Nhiều ant → explore tốt hơn nhưng chậm hơn

    double alpha = 1.0;                                 // hệ số importance của PHEROMONE
                                                        // alpha lớn → kiến ưu tiên đường có nhiều pheromone
                                                        // (exploitation > exploration)

    double beta = 1.0;                                  // hệ số importance của HEURISTIC (desirability)
                                                        // beta lớn → kiến ưu tiên cluster "tốt" về distance + capacity
                                                        // (greedy hơn)

    double rho = 0.2;                                   // tỷ lệ bay hơi pheromone (evaporation rate)
                                                        // Mỗi iteration: phi *= (1-rho)
                                                        // rho lớn → quên nhanh → explore nhiều
                                                        // rho nhỏ → nhớ lâu → exploit nhiều

    // ── Tham số adaptive (thay đổi theo quá trình chạy) ──

    double T_max = 0.3;                                 // lượng pheromone deposit tối đa (best ant)
    double T_min = 0.1;                                 // lượng pheromone deposit tối thiểu (all ants)
                                                        // T_max >> T_min → best ant ảnh hưởng mạnh
    const double PHI_MIN = 0.1;                         // giới hạn dưới vết mùi (tránh = 0)
    const double PHI_MAX = 1.0;                         // giới hạn trên vết mùi (tránh quá lớn → bias cực)

    double Q_max = 0.95;                                // xác suất exploitation tối đa
    double Q_min = 0.05;                                // xác suất exploitation tối thiểu
    double Q0 = Q_max;                                  // xác suất exploitation hiện tại
                                                        // Q0 cao → kiến thường chọn cluster tốt nhất (exploit)
                                                        // Q0 thấp → kiến chọn theo roulette wheel (explore)

    // Stagnation parameters
    int STAGNATE_LIMIT = 1;                             // sau bao nhiêu iteration không improve → bắt đầu giảm Q_max
                                                        // → chuyển dần từ exploitation sang exploration
    int STAGNATE_COUNT = 0;                             // đếm số vòng liên tiếp không cải thiện sau khi giảm Q_0

    int STAGNATE_DROP = 0.05;                           // Ý định: mỗi iteration stagnate, giảm Q0 đi 0.05

    // ── Repair configuration ──
    int lsTop = 10;                                     // số ant được chọn để local search
                                                        // Không repair TẤT CẢ m ant (quá chậm)
                                                        // Chỉ repair top ants (theo cost) + đa dạng (Hamming)
    int lsMaxMoves = 1000;                              // giới hạn moves mỗi lần LS

    // Số ant được Tabu Search (subset của lsTop)
    int tsTop = 5;                                     // số ant được chọn để tabu search

    // Điều kiện chạy Tabu Search
    int tsInterval = 10;                                // chạy TS mỗi x iteration

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
    
    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 7: KHỞI TẠO ANT POOL — ELITE ARCHIVE LIÊN ITERATION
    //
    // antPool: lưu trữ các nghiệm tốt + đa dạng từ NHIỀU iteration trước.
    // Thay vì chỉ chọn tsTop kiến trong iteration hiện tại để chạy TS,
    // ta tích lũy pool qua nhiều iteration rồi chọn ứng viên TS từ đó.
    //
    // Lợi ích:
    //   - Tránh chạy TS trên các kiến quá giống nhau (cùng iteration)
    //   - Pool chứa nhiều "local optima" từ các vùng khác nhau → TS explore
    //     nhiều basin of attraction hơn
    //   - Sau reset pheromone, pool vẫn giữ lại nghiệm tốt cũ → không mất
    //
    // Chiến lược duy trì pool:
    //   Khi thêm ant mới vào pool:
    //     (1) Nếu pool chưa đầy → thêm thẳng
    //     (2) Nếu pool đầy:
    //         - Loại ant tệ nhất TRONG POOL nếu ant mới tốt hơn VÀ đủ đa dạng
    //         - Nếu ant mới quá giống 1 ant đã có → chỉ thay thế nếu tốt hơn ant đó
    //
    // poolSize  = tsInterval * tsTop: tích lũy đủ để có ứng viên từ nhiều iteration
    // MIN_DIFF_POOL: Hamming distance tối thiểu giữa các phần tử trong pool
    // ─────────────────────────────────────────────────────────────────────
 
    const int poolSize     = max(tsInterval * tsTop, 1);         // kích thước pool tối đa
    const int MIN_DIFF_POOL = max(N / 10, 1);            // độ đa dạng tối thiểu trong pool
 
    struct PoolEntry {
        ACOSolution sol;
    };
    vector<PoolEntry> antPool;                           // elite archive
    antPool.reserve(poolSize);
 
    // Hàm thêm ant vào pool với chiến lược diversity-aware replacement
    auto addToPool = [&](const ACOSolution &candidate) {
        // Tìm xem có ant nào trong pool quá giống candidate không
        int twinIdx = -1;      // index ant trong pool giống candidate nhất
        int twinDist = max(N/100, 1);  // Hamming dist tới twin
        for (int pi = 0; pi < (int)antPool.size(); ++pi) {
            int hd = hamming_distance(candidate.assign, antPool[pi].sol.assign);
            if (hd < twinDist) { twinDist = hd; twinIdx = pi; }
        }
 
        if ((int)antPool.size() < poolSize) {
            // Pool chưa đầy → thêm thẳng (dù có twin)
            // Nếu có twin quá gần (< MIN_DIFF_POOL) → chỉ giữ cái tốt hơn
            if (twinIdx >= 0 && twinDist < MIN_DIFF_POOL) {
                if (candidate.cost < antPool[twinIdx].sol.cost - 1e-9) {
                    antPool[twinIdx].sol       = candidate;
                }
                // Nếu twin tốt hơn → bỏ qua candidate (không thêm duplicate)
            } else {
                antPool.push_back({candidate});
            }
        } else {
            // Pool đầy → cần thay thế
            if (twinIdx >= 0 && twinDist < MIN_DIFF_POOL) {
                // Có twin trong pool: chỉ swap nếu candidate tốt hơn twin
                if (candidate.cost < antPool[twinIdx].sol.cost - 1e-9) {
                    antPool[twinIdx].sol       = candidate;
                }
            } else {
                // Candidate đủ đa dạng → thay thế ant tệ nhất trong pool
                int worstIdx  = 0;
                double worstCost = antPool[0].sol.cost;
                for (int pi = 1; pi < (int)antPool.size(); ++pi) {
                    if (antPool[pi].sol.cost > worstCost) {
                        worstCost = antPool[pi].sol.cost;
                        worstIdx  = pi;
                    }
                }
                if (candidate.cost < worstCost - 1e-9) {
                    antPool[worstIdx].sol       = candidate;
                }
            }
        }
    };

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
                    double deltaDist = ants[a].clusterSumDist[i][k];

                    double fitness = 0.0;

                    for (int t = 0; t < M_weights; ++t)
                    {
                        double cur   = ants[a].clusterWeight[k][t];
                        double w     = Wmat[i][t];
                        double after = cur + w;

                        double lo   = WLmat[k][t];
                        double hi   = WUmat[k][t];
                        double span = max(hi - lo, VALID_EPS);

                        double fitness_t = 0.0;

                        if (after > hi)
                        {
                            // ── Vi phạm upper bound ──
                            // fitness bắt đầu tại 0.3, giảm chậm khi vi phạm tăng.
                            // Dùng log để phạt nhẹ: vi phạm gấp đôi span chỉ mất ~0.1 fitness.
                            // Local search sẽ sửa → không cần phạt quá nặng ở đây.
                            double over = (after - hi) / span;      // vi phạm tính theo span
                            fitness_t = 3.0 / (1.0 + log1p(over));  // [0.3 → 0 chậm]
                        }
                        else if (after < lo)
                        {
                            double room_ratio = after / max(lo,VALID_EPS);               // tỉ lệ w dùng vào phần thiếu
                            fitness_t = 0.75 + 0.25 * room_ratio; // [0.75, 1.0]
                        }
                        else
                        {
                            // ── Cluster đã thỏa mãn [lo, hi], thêm node vẫn trong giới hạn ──
                            // Không "cần thiết" nhưng cũng không hại.
                            // fitness ≈ 0.7, giảm nhẹ khi gần chạm upper (ít room hơn).
                            double room_ratio = (hi - after) / span;   // 1.0 = rất thoáng, 0.0 = sát upper
                            fitness_t = 0.6 + 0.15 * room_ratio;      // [0.50, 0.75]
                        }

                        fitness += log(max(fitness_t, 1e-12));
                    }

                    // Trung bình fitness qua tất cả các chiều trọng số
                    fitness = exp(fitness / M_weights);;

                    // ── Desirability = fitness / distance (normalized) ──
                    // Thêm offset 0.5*DIST_SCALE tránh div-by-zero khi cluster rỗng.
                    // Cluster rỗng: dist_term = 0.5 → desir = 2 * fitness (thưởng nhẹ để khuyến khích fill cluster mới)
                    double dist_term = (deltaDist + DIST_SCALE * 0.5) / DIST_SCALE;
                    double desir = fitness / dist_term;

                    double tau    = phi[i][k];
                    double weight = pow(tau, alpha) * pow(desir, beta);

                    weights[k] = weight;

                    if (weight > bestWeight)
                    {
                        bestWeight = weight;
                        chosenK    = k;
                    }
                }
                
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
                // Khi stagnate → Q0 giảm → explore nhiều hơn
                // ═════════════════════════════════════════════════════════

                double q = uni01(rng);                    // random trong [0, 1)

                if (q >= Q0)                          // q >= antQ0 → EXPLORATION (roulette wheel)
                                                         // Logic: q >= antQ0 → explore
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
                // Nếu q < antQ0 → EXPLOITATION → giữ chosenK = argmax(weights)

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
            ants[a].cost = compute_cost_fast(ants[a]);
            ants[a].feasible = is_feasible(ants[a].assign);

        } // ── Kết thúc construction phase (m ants built) ──

        // ═════════════════════════════════════════════════════════════════
        // PHASE 2: IMPROVEMENT — Local Search + Tabu Search cho top ants
        //
        // Không improve TẤT CẢ m ants (quá chậm).
        // Chọn top ants TỐT NHẤT + ĐA DẠNG:
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

        // ── Chọn top ants đa dạng ──
        vector<int> lsSelected;                          // index các ant được chọn để local search
        int MIN_DIFF_LS = max(N/100,1);                     // Hamming distance tối thiểu giữa các ant selected

        for (int idx = 0; idx < m && (int)lsSelected.size() < lsTop; ++idx)
        {
            int ai = order[idx];                         // ant tốt thứ idx
            bool diverse = true;                         // giả sử đủ khác biệt

            // Kiểm tra Hamming distance với tất cả ant đã chọn
            for (int sj : lsSelected)
            {
                if (hamming_distance(ants[ai].assign, ants[sj].assign) < MIN_DIFF_LS)
                {
                    diverse = false;                     // quá giống ant đã chọn → bỏ qua
                    break;
                }
            }

            if (diverse)
                lsSelected.push_back(ai);                // thêm vào danh sách repair
        }

        // Nếu chưa đủ repairTop ant (do tất cả quá giống nhau)
        // → bổ sung thêm ant theo thứ tự cost (không check Hamming)
        for (int idx = 0; idx < m && (int)lsSelected.size() < lsTop; ++idx)
        {
            int ai = order[idx];
            if (find(lsSelected.begin(), lsSelected.end(), ai) == lsSelected.end())
                lsSelected.push_back(ai);                // thêm vào
        }

        // ── LOCAL SEARCH cho tất cả ant được chọn ──
        //   LS nhanh → chạy cho nhiều ant → cải thiện population chung
        for (int ai : lsSelected)
        {
            local_search(ants[ai], rng, lsMaxMoves);

            ants[ai].cost = compute_cost_fast(ants[ai]);
            ants[ai].feasible = is_feasible(ants[ai].assign);
        }

        // ═════════════════════════════════════════════════════════════
        // NẠP POOL SAU LOCAL SEARCH
        //
        // Sau mỗi iteration, thêm các ant vừa được LS vào antPool.
        // Pool tích lũy dần qua nhiều iteration → khi đến lượt chạy TS
        // sẽ có nhiều ứng viên chất lượng cao từ nhiều vùng tìm kiếm.
        //
        // Chỉ nạp lsSelected (đã qua LS) — không nạp ant thô từ construction
        // vì chúng chưa được cải thiện và thường có cost cao.
        // ═════════════════════════════════════════════════════════════
        for (int ai : lsSelected)
            addToPool(ants[ai]);
        
        if (iter % tsInterval == 0) // chay tabu search mỗi tsInterval vòng
        {
            // ═════════════════════════════════════════════════════════════
            // CHỌN ỨNG VIÊN TS TỪ POOL LIÊN ITERATION
            // Chọn tsTop kiến TỐT NHẤT + ĐA DẠNG từ antPool —
            // pool tích lũy từ nhiều iteration trước (tối đa poolSize phần tử).
            //
            // Sort pool theo cost, chọn top đa dạng.
            // ═════════════════════════════════════════════════════════════
    
            // Sort pool theo cost tăng dần
            sort(antPool.begin(), antPool.end(),
                [](const PoolEntry &a, const PoolEntry &b){
                    return a.sol.cost < b.sol.cost;
                });
    
            // Chọn tsTop phần tử đa dạng từ pool
            vector<int> tsPoolSelected;
    
            for (int pi = 0; pi < (int)antPool.size() && (int)tsPoolSelected.size() < tsTop; ++pi)
            {
                tsPoolSelected.push_back(pi);
            }
    
            // ── Chạy Iterated Tabu Search trên ứng viên từ pool ──
            for (int pi : tsPoolSelected)
            {
                // antPool[pi].sol là nghiệm đã qua LS từ iteration trước
                // → TS khai thác sâu hơn + thoát local opt
                iterated_tabu_search(antPool[pi].sol, rng);
    
                antPool[pi].sol.cost     = compute_cost_fast(antPool[pi].sol);
                antPool[pi].sol.feasible = is_feasible(antPool[pi].sol.assign);
    
                // Nghiệm sau TS cũng là ứng viên để cập nhật best global
                bool curFeasible  = antPool[pi].sol.feasible;
                double curCost    = antPool[pi].sol.cost;

                bool bestFeasible = best.feasible;
                double bestCost   = best.cost;

                bool accept = false;

                if (curFeasible)
                {
                    if (!bestFeasible || curCost + VALID_EPS < bestCost)
                        accept = true;
                }
                else
                {
                    if (!bestFeasible && curCost + VALID_EPS < bestCost)
                        accept = true;
                }

                if (accept)
                {
                    best = antPool[pi].sol;

                    auto now = Clock::now();
                    double elapsed = chrono::duration<double>(now - start).count();

                    noImprove = 0;                           // reset counter stagnation
                    STAGNATE_COUNT = 0;
                    Q0 = Q_max;                             // reset Q_max

                    cerr << "[ITER " << iter << "]"
                        << " [TS] New best cost=" 
                        << format_cost_with_commas(best.cost, 2)
                        << " (feasible=" << (best.feasible ? "YES" : "NO")
                        << ", time " << elapsed << "s)\n";
                }
            }

            antPool.clear();
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
                if (!bestFeasible || curCost + VALID_EPS < bestCost)
                    accept = true;
            }
            else
            {
                // Trường hợp 2: Ant mới INFEASIBLE
                //   → Chỉ chấp nhận nếu best cũng infeasible VÀ cost nhỏ hơn
                //   → KHÔNG BAO GIỜ thay thế best feasible bằng infeasible
                if (!bestFeasible && curCost + VALID_EPS < bestCost)
                    accept = true;
            }

            if (accept)
            {
                best = ants[ai];                         // cập nhật best global
                improvedThisIter = true;                 // đánh dấu iteration này có cải thiện

                noImprove = 0;                           // reset counter stagnation
                STAGNATE_COUNT = 0;
                Q0 = Q_max;                             // reset Q_max

                // In thông báo ra stderr
                auto now = Clock::now();
                double elapsed = chrono::duration<double>(now - start).count();
                cerr << "[ITER " << iter << "] New best cost=" << format_cost_with_commas(best.cost, 2)
                     << " (feasible=" << (best.feasible ? "YES" : "NO")
                     << ", time " << elapsed << "s)\n";
            }
        }

        // Nếu iteration này KHÔNG cải thiện best
        if (!improvedThisIter)
        {
            ++noImprove;                                 // tăng counter stagnation
            ++STAGNATE_COUNT;
            if(STAGNATE_COUNT >= STAGNATE_LIMIT){
                Q0 -= STAGNATE_DROP;
                STAGNATE_COUNT = 0;
                if (Q0 < Q_min) Q0 = Q_min; // Q_max limit
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

        // Bước 3: Deposit từ ITERATION BEST (khác global)
        int bestLocal = -1;
        int MIN_DIFF = max(N/100,1);
        for (int r = 0; r < m; ++r) {
            int ai = order[r];
            if (hamming_distance(ants[ai].assign, best.assign) >= MIN_DIFF) {
                bestLocal = ai;
                break;
            }
        }

        if (bestLocal >= 0) {
            // Deposit iteration-best với trọng số nhỏ hơn global
            double iterDeposit = T_max * 0.3;
            for (int i = 0; i < N; ++i) {
                int c = ants[bestLocal].assign[i];
                if (c >= 0 && c < K)
                    phi[i][c] += iterDeposit;
            }
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
            // Best local ant
            auto bestThisIter = ants[order[0]];

            // Đếm số ant feasible trong iteration này
            int feasCount = 0;
            for (int a = 0; a < m; ++a)
                if (ants[a].feasible)
                    feasCount++;

            // In ra stderr
            double elapsed = chrono::duration<double>(Clock::now() - start).count();
            cerr << "[ITER " << iter << "] bestGlobalCost=" << format_cost_with_commas(best.cost, 2)
                 << " (feasible=" << (best.feasible ? "YES" : "NO") << ")"
                 << " bestThisIter=" << format_cost_with_commas(bestThisIter.cost, 2)
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
        //   → Cho phép ACO "khám phá lại từ đầu" với kinh nghiệm mới
        //
        // Đây là cơ chế thoát local optima quan trọng nhất.
        // ═════════════════════════════════════════════════════════════════

        int noImproveReset = 30;  // ngưỡng reset (50 iteration)

        if (noImprove >= noImproveReset)
        {
            cerr << "[RESET] no improvement for" << noImproveReset << "-> reset pheromones\n";

            // Reset toàn bộ pheromone về T_min
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < K; ++k)
                    phi[i][k] = T_min;                   // phi = 0.05 (uniform)

            noImprove = 0;                               // reset counter
            STAGNATE_DROP = 0;
            Q0 = Q_max;
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

    cout << "Final cost = " << format_cost_with_commas(best.cost, 2) << "\n";

    // Ghi tất cả log ra file
    SaveLogs(best);

    return best;                                         // trả nghiệm tốt nhất
}
