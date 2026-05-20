// ═══════════════════════════════════════════════════════════════════════════
// FILE: Tabu_search.cpp
//
// THUẬT TOÁN: Iterated Tabu Search (ITS) — Tìm kiếm Tabu Lặp
//
// MỤC TIÊU:
//   Tối thiểu hóa tổng khoảng cách nội cụm (intra-cluster distance) trong
//   bài toán phân cụm có ràng buộc trọng số (Capacitated Clustering Problem).
//   Mỗi cluster k phải thỏa: WLmat[k][t] ≤ tổng trọng số chiều t ≤ WUmat[k][t].
//
// Ý TƯỞNG CHÍNH:
//   - Tabu Search (TS): tìm kiếm cục bộ, cho phép đi sang trạng thái xấu hơn
//     để thoát khỏi local optimum, dùng "danh sách cấm" để tránh quay vòng.
//   - Iterated Local Search (ILS): lặp lại TS sau mỗi lần "phá vỡ" (perturbation)
//     nghiệm để khám phá các vùng khác của không gian tìm kiếm.
//
// LUỒNG THỰC HIỆN TỔNG QUÁT:
//   1. TS warm-up: chạy TS trên nghiệm đầu vào để khai thác local opt sâu hơn.
//   2. Vòng lặp ILS:
//      a. Perturbation: di chuyển ngẫu nhiên một số đỉnh sang cluster khác.
//      b. TS focused: chỉ tìm kiếm trong vùng vừa bị perturbation.
//      c. TS full-pass: tìm kiếm toàn bộ N đỉnh để xử lý hiệu ứng lan rộng.
//      d. Acceptance: chấp nhận nghiệm mới nếu đủ tốt (có cửa sổ chấp nhận).
//      e. Restart: nếu bị tắc nghẽn quá lâu, quay lại nghiệm tốt nhất + perturbation mạnh.
//   3. Trả về nghiệm tốt nhất tìm được.
//
// HAI LOẠI MOVE:
//   - RELOCATE: di chuyển 1 đỉnh u từ cluster 'from' sang cluster 'to'.
//   - SWAP: hoán đổi 2 đỉnh u, v đang ở 2 cluster khác nhau.
//
// TABU LIST:
//   tabu[u][k] = bước tuyệt đối X: đỉnh u không được RỜI cluster k cho đến bước X.
//   → Sau khi di chuyển u: from→to, ta set tabu[u][from] để cấm reverse move.
//   → Tabu list được dùng chung (shared) xuyên suốt toàn bộ ITS.
//
// DYNAMIC PENALTY:
//   Nghiệm có thể vi phạm ràng buộc trọng số (infeasible).
//   Hàm mục tiêu = intra_dist + pen × total_violation.
//   'pen' tăng dần khi nghiệm liên tục infeasible → đẩy về vùng feasible.
//   'pen' giảm khi feasible + bị kẹt → mở rộng không gian thoát local opt.
// ═══════════════════════════════════════════════════════════════════════════

#include "ACO.h"          // Chứa cấu trúc ACOSolution, hàm compute_cost_fast, is_feasible
#include "Tabu_search.h"  // Khai báo hàm iterated_tabu_search
#include "Local_search.h" // Các hằng số: PENALTY_SCALE, VALID_EPS, và ma trận toàn cục
#include <algorithm>      // std::find, std::sort, std::shuffle, std::rotate, std::partial_sort
#include <numeric>        // std::iota (điền giá trị 0,1,2,...)
#include <cmath>          // std::max, std::min
#include <random>         // std::mt19937_64, std::uniform_int_distribution, v.v.

// Biến toàn cục được định nghĩa ở file khác (dùng extern để tham chiếu):
// globalCL: Candidate List — danh sách các đỉnh gần nhất với mỗi đỉnh
// GLOBAL_CL_SIZE: kích thước của mỗi danh sách ứng viên
extern vector<vector<int>> globalCL;
extern int                 GLOBAL_CL_SIZE;


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 1: HÀM HELPER — ts_relocate và ts_swap
//
// Hai hàm này thực hiện việc di chuyển đỉnh và cập nhật toàn bộ cấu trúc
// dữ liệu của nghiệm (assign, members, clusterWeight, clusterSumDist).
//
// Cập nhật incremental (không tính lại từ đầu) để đảm bảo hiệu năng O(N).
// ═══════════════════════════════════════════════════════════════════════════

// ts_erase: xóa một phần tử khỏi vector bằng cách swap với phần tử cuối rồi pop_back.
// Phức tạp O(N) cho find, O(1) cho xóa — nhanh hơn erase(it) vì không dịch chuyển phần tử.
static inline void ts_erase(vector<int> &v, int node) {
    auto it = find(v.begin(), v.end(), node); // Tìm vị trí của 'node' trong vector
    if (it != v.end()) {
        *it = v.back(); // Ghi đè phần tử cần xóa bằng phần tử cuối
        v.pop_back();   // Xóa phần tử cuối (đã được sao chép lên trên)
    }
}

// ts_relocate: di chuyển đỉnh u từ cluster 'from' sang cluster 'to'.
// Cập nhật 4 thành phần của ACOSolution:
//   1. sol.assign[u]: nhãn cluster của u
//   2. sol.members[from/to]: danh sách thành viên của cluster
//   3. sol.clusterWeight[from/to]: tổng trọng số mỗi chiều của cluster
//   4. sol.clusterSumDist[v][from/to]: tổng khoảng cách từ v đến các thành viên cluster
static void ts_relocate(ACOSolution &sol, int u, int from, int to)
{
    // --- Cập nhật nhãn cluster ---
    sol.assign[u] = to;         // u bây giờ thuộc cluster 'to'
    ts_erase(sol.members[from], u);  // Xóa u khỏi danh sách thành viên của 'from'
    sol.members[to].push_back(u);    // Thêm u vào danh sách thành viên của 'to'

    // --- Cập nhật tổng trọng số cluster ---
    // Duyệt qua M_weights chiều trọng số (M_weights là hằng số toàn cục)
    for (int t = 0; t < M_weights; ++t) {
        sol.clusterWeight[from][t] -= Wmat[u][t]; // 'from' mất trọng số của u
        sol.clusterWeight[to][t]   += Wmat[u][t]; // 'to' nhận trọng số của u
    }

    // --- Cập nhật clusterSumDist ---
    // clusterSumDist[v][k] = tổng dist(v, m) với mọi m ∈ cluster k
    // Khi u chuyển from→to: mọi đỉnh v ≠ u đều bị ảnh hưởng:
    //   clusterSumDist[v][from] giảm đi dist(v,u)
    //   clusterSumDist[v][to]   tăng lên dist(v,u)
    for (int v = 0; v < N; ++v) {
        if (v == u) continue;      // Bỏ qua chính u (xử lý riêng bên dưới)
        double d = distmat[v][u];  // Khoảng cách từ v đến u
        sol.clusterSumDist[v][from] -= d; // v thấy 'from' gần hơn (mất u)
        sol.clusterSumDist[v][to]   += d; // v thấy 'to' xa hơn (có thêm u)
    }

    // Tính lại clusterSumDist[u][from] (tổng khoảng cách từ u đến thành viên 'from' CŨ)
    // Sau khi u đã rời 'from', 'from' không còn u nên tính trực tiếp từ danh sách mới
    { double s = 0.0;
      for (int m : sol.members[from]) s += distmat[u][m]; // Tổng dist(u, m) ∀m ∈ from hiện tại
      sol.clusterSumDist[u][from] = s; }

    // Tính lại clusterSumDist[u][to] (tổng khoảng cách từ u đến thành viên 'to' MỚI)
    // 'to' đã bao gồm u rồi, nhưng distmat[u][u] = 0 nên không ảnh hưởng
    { double s = 0.0;
      for (int m : sol.members[to])   s += distmat[u][m]; // Tổng dist(u, m) ∀m ∈ to hiện tại
      sol.clusterSumDist[u][to]   = s; }
}

// ts_swap: hoán đổi 2 đỉnh u và v đang ở 2 cluster khác nhau (cu ≠ cv).
// u chuyển từ cu sang cv, v chuyển từ cv sang cu.
// Cập nhật tương tự ts_relocate nhưng cho 2 đỉnh cùng lúc.
static void ts_swap(ACOSolution &sol, int u, int v)
{
    int cu = sol.assign[u]; // Cluster hiện tại của u
    int cv = sol.assign[v]; // Cluster hiện tại của v
    if (cu == cv) return;   // Cùng cluster → swap không có ý nghĩa, bỏ qua

    // --- Cập nhật nhãn cluster ---
    sol.assign[u] = cv; sol.assign[v] = cu; // u↔v hoán đổi cluster
    ts_erase(sol.members[cu], u); ts_erase(sol.members[cv], v); // Xóa khỏi cluster cũ
    sol.members[cv].push_back(u); sol.members[cu].push_back(v); // Thêm vào cluster mới

    // --- Cập nhật tổng trọng số ---
    // cu nhận trọng số v, mất trọng số u (vì u đi v đến)
    // cv nhận trọng số u, mất trọng số v (vì v đi u đến)
    for (int t = 0; t < M_weights; ++t) {
        sol.clusterWeight[cu][t] += Wmat[v][t] - Wmat[u][t]; // cu: +v -u
        sol.clusterWeight[cv][t] += Wmat[u][t] - Wmat[v][t]; // cv: +u -v
    }

    // --- Cập nhật clusterSumDist cho mọi đỉnh w ≠ u, v ---
    // Khi u: cu→cv và v: cv→cu:
    //   clusterSumDist[w][cu] += dist(w,v) - dist(w,u)  (cu mất u, được v)
    //   clusterSumDist[w][cv] += dist(w,u) - dist(w,v)  (cv mất v, được u)
    for (int w = 0; w < N; ++w) {
        if (w == u || w == v) continue; // u và v xử lý riêng bên dưới
        double du = distmat[w][u]; // Khoảng cách từ w đến u
        double dv = distmat[w][v]; // Khoảng cách từ w đến v
        sol.clusterSumDist[w][cu] += dv - du; // cu mất u (+d(w,v)) bù lại bằng v
        sol.clusterSumDist[w][cv] += du - dv; // cv mất v (+d(w,u)) bù lại bằng u
    }

    // Tính lại clusterSumDist[u][k] và clusterSumDist[v][k] cho k ∈ {cu, cv}
    // Sau khi hoán đổi, u ở cv và v ở cu, cần tính lại tổng khoảng cách
    for (int k : {cu, cv}) {
        double su = 0.0, sv = 0.0;
        for (int m : sol.members[k]) {
            su += distmat[u][m]; // Tổng khoảng cách từ u đến mọi thành viên của k
            sv += distmat[v][m]; // Tổng khoảng cách từ v đến mọi thành viên của k
        }
        sol.clusterSumDist[u][k] = su;
        sol.clusterSumDist[v][k] = sv;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 2: TS CORE — Một phiên Tìm kiếm Tabu
//
// Hàm cốt lõi của thuật toán. Mỗi lần gọi chạy 'maxSteps' bước TS.
//
// MỖI BƯỚC TS:
//   Phase A (RELOCATE):
//     - Duyệt hết nodeOrder (N đỉnh hoặc tập con nếu focusedNodes được cung cấp).
//     - Với mỗi đỉnh u, xét tất cả cluster 'to' ≠ cluster hiện tại.
//     - Tính score = deltaDist + pen × deltaViol cho mỗi move.
//     - Theo dõi 2 ứng viên: (A) move cải thiện tốt nhất, (B) move fallback (ít xấu nhất).
//     - Sau khi duyệt hết: apply (A) nếu có, bỏ qua Phase B; nếu không có (A) mới thử (B).
//   Phase B (SWAP):
//     - Chỉ chạy khi Phase A không tìm được move cải thiện.
//     - Dùng first-improvement: apply ngay khi tìm thấy move cải thiện đầu tiên.
//     - Thử swap với neighbor trong CL trước, sau đó swap ngẫu nhiên.
//   FALLBACK:
//     - Nếu cả Phase A lẫn B đều không cải thiện → apply move ít xấu nhất (fallback).
//     - Đây là đặc điểm cốt lõi của TS: chấp nhận trạng thái xấu hơn để thoát local opt.
//
// TABU LIST:
//   tabu[u][k] = bước tuyệt đối X → u không được RỜI cluster k đến bước X.
//   "Bước tuyệt đối" = stepOffset + step_hiện_tại (để tenure đúng qua nhiều lần gọi).
//   Sau khi apply move u: from→to → set tabu[u][from] = absStep + tenure.
//   Aspiration criterion: override tabu nếu nghiệm mới feasible VÀ tốt hơn bestFeas.
// ═══════════════════════════════════════════════════════════════════════════

static void ts_core(
    ACOSolution               &sol,          // Nghiệm đang làm việc (bị sửa trực tiếp)
    mt19937_64                &rng,          // Bộ sinh số ngẫu nhiên
    int                        maxSteps,     // Số bước TS tối đa
    ACOSolution               &bestFeasSol,  // Nghiệm khả thi tốt nhất đã tìm (shared)
    double                    &bestFeasIntra,// Intra-dist của bestFeasSol (shared)
    bool                      &hasBestFeas, // Đã tìm được nghiệm khả thi chưa (shared)
    int                        tabuBase,    // Tenure cơ bản của tabu list
    int                        tabuDelta,   // Biên độ ngẫu nhiên của tenure
    const vector<vector<int>> &cl,          // Candidate List: cl[u] = các đỉnh gần u nhất
    vector<vector<int>>       &tabu,        // Tabu list shared (tabu[u][k] = bước cấm)
    int                        stepOffset,  // Tổng số bước đã chạy trước lần gọi này
    const vector<int>         *focusedNodes = nullptr) // Nếu không null: chỉ duyệt tập con này
{
    const double EPS       = 1e-9; // Ngưỡng so sánh số thực (tránh lỗi floating point)
    const double SCORE_EPS = 1e-9; // Ngưỡng để xác định "move có cải thiện" (score < -SCORE_EPS)

    // Hàm tính "bước tuyệt đối" tại step s của lần gọi này.
    // stepOffset là tổng bước đã chạy trước đó, đảm bảo tenure so sánh đúng
    // khi ts_core được gọi nhiều lần liên tiếp với tabu list chung.
    auto absStep = [&](int s) -> int { return stepOffset + s; };

    // ── Tính vi phạm ràng buộc trọng số cho mỗi cluster ──
    // vc[k] = tổng lượng vi phạm của cluster k trên tất cả chiều trọng số.
    // Vi phạm = max(0, weight - upper_bound) + max(0, lower_bound - weight).
    auto computeClViol = [&](int k) -> double {
        double v = 0.0;
        for (int t = 0; t < M_weights; ++t) {
            double s = sol.clusterWeight[k][t]; // Tổng trọng số chiều t của cluster k
            if (s > WUmat[k][t]) v += s - WUmat[k][t]; // Vượt upper bound
            if (s < WLmat[k][t]) v += WLmat[k][t] - s; // Dưới lower bound
        }
        return v;
    };

    vector<double> vc(K); // vi phạm của từng cluster (K cluster tổng cộng)
    double totalViol = 0.0;
    for (int k = 0; k < K; ++k) {
        vc[k]      = computeClViol(k); // Tính vi phạm ban đầu cho cluster k
        totalViol += vc[k];            // Cộng dồn vào tổng vi phạm
    }
    totalViol = max(totalViol, 0.0); // Đảm bảo không âm (lỗi floating point)

    // ── Tính intra-distance hiện tại ──
    // Intra-distance = tổng khoảng cách giữa mỗi đỉnh i và tất cả thành viên
    // trong cùng cluster với i. Đây là giá trị ta muốn minimize.
    double curIntra = 0.0;
    for (int i = 0; i < N; ++i)
        curIntra += sol.clusterSumDist[i][sol.assign[i]]; // dist(i → các thành viên cùng cluster)

    // Cập nhật bestFeas ngay nếu nghiệm đầu vào đã feasible
    if (totalViol < VALID_EPS) { // VALID_EPS: ngưỡng coi là feasible (gần 0)
        if (!hasBestFeas || curIntra < bestFeasIntra - EPS) {
            hasBestFeas   = true;
            bestFeasIntra = curIntra;
            bestFeasSol   = sol;
        }
    }

    // ── Dynamic penalty ──
    // pen: hệ số phạt vi phạm ràng buộc trong hàm mục tiêu.
    // score = deltaDist + pen × deltaViol
    //
    // Khi infeasible: tăng pen dần theo chu kỳ gradSteps để đẩy về feasible.
    // Khi feasible + bị kẹt (phải dùng fallback): giảm pen để mở rộng tìm kiếm.
    double pen = PENALTY_SCALE;                      // Giá trị khởi đầu (từ Local_search.h)
    const double PEN_UP     = 1.5;                   // Hệ số nhân khi thay đổi pen
    const double PEN_DOWN   = 1.5;                   // Hệ số chia khi thay đổi pen
    const double PEN_MIN    = PENALTY_SCALE * 0.3;   // Giới hạn dưới của pen
    const double PEN_MAX    = PENALTY_SCALE * 3.0;   // Giới hạn trên của pen
    const int    gradSteps  = 5;                     // Số bước infeas liên tiếp trước khi tăng pen
    int stepsInfeas = 0;                             // Đếm số bước infeas liên tiếp

    // updatePen: gọi sau mỗi move được apply.
    // Nếu feasible → giữ nguyên pen (không cần thêm áp lực).
    // Nếu infeasible → tăng pen sau mỗi gradSteps bước liên tiếp.
    auto updatePen = [&]() {
        if (totalViol < VALID_EPS) {
            stepsInfeas = 0; // Đang feasible → reset đếm, giữ pen
        } else {
            ++stepsInfeas;
            if (stepsInfeas >= gradSteps) {
                pen         = min(pen * PEN_UP, PEN_MAX); // Tăng pen, không vượt PEN_MAX
                stepsInfeas = 0; // Reset đếm sau mỗi lần tăng
            }
        }
    };

    // updatePenIdle: gọi khi bước TS phải dùng fallback (bị kẹt).
    // Thay đổi penalty NGAY LẬP TỨC (không chờ gradSteps) vì đây là tín hiệu
    // rõ ràng rằng landscape hiện tại đã cạn kiệt move cải thiện.
    //   - Feasible + kẹt → giảm pen ngay để mở cửa sổ không gian tìm kiếm.
    //   - Infeasible + kẹt → tăng pen ngay để ép mạnh về feasible.
    auto updatePenIdle = [&]() {
        if (totalViol < VALID_EPS) {
            pen         = max(pen / PEN_DOWN, PEN_MIN); // Giảm pen ngay, không dưới PEN_MIN
            stepsInfeas = 0;
        } else {
            pen         = min(pen * PEN_UP, PEN_MAX);   // Tăng pen ngay, không vượt PEN_MAX
            stepsInfeas = 0;
        }
    };

    // ── Phân phối ngẫu nhiên cho tenure và chọn đỉnh ngẫu nhiên ──
    uniform_int_distribution<int> tenureRand(-tabuDelta, tabuDelta); // Tenure = tabuBase ± tabuDelta
    uniform_int_distribution<int> randNodeDist(0, N - 1);           // Chọn đỉnh ngẫu nhiên trong [0, N-1]

    // ── Xây dựng thứ tự duyệt đỉnh ──
    // nodeOrder: thứ tự duyệt cho Phase A (RELOCATE) — có thể là tập con
    // fullOrder: thứ tự duyệt cho Phase B (SWAP) — luôn là toàn bộ N đỉnh
    vector<int> nodeOrder;
    vector<int> fullOrder(N);
    iota(fullOrder.begin(), fullOrder.end(), 0); // fullOrder = {0, 1, 2, ..., N-1}

    if (focusedNodes && !focusedNodes->empty()) {
        nodeOrder = *focusedNodes; // Dùng tập con được cung cấp (TS focused)
    } else {
        nodeOrder = fullOrder;     // Dùng toàn bộ N đỉnh (TS full-pass)
    }
    shuffle(nodeOrder.begin(), nodeOrder.end(), rng); // Shuffle để tránh bias thứ tự duyệt
    shuffle(fullOrder.begin(), fullOrder.end(), rng); // Shuffle cho SWAP phase

    // Ngưỡng aspiration: nếu một move tabu dẫn đến nghiệm feasible + tốt hơn ngưỡng này
    // thì được phép thực hiện dù bị cấm (aspiration criterion).
    double aspireFeas = hasBestFeas ? bestFeasIntra : 1e300;

    // ══════════════════════════════════════════════════════════════════════
    // VÒNG LẶP CHÍNH TS CORE: chạy maxSteps bước
    // ══════════════════════════════════════════════════════════════════════
    for (int step = 0; step < maxSteps; ++step)
    {
        // Rotate nodeOrder 1 vị trí: đỉnh đầu tiên → cuối cùng.
        // Cách đơn giản để thay đổi thứ tự duyệt mỗi bước mà không cần shuffle O(N).
        rotate(nodeOrder.begin(), nodeOrder.begin() + 1, nodeOrder.end());

        // ══════════════════════════════════════════════════════════════════
        // PHASE A — RELOCATE: Tìm move tốt nhất trong toàn nodeOrder
        //
        // Duyệt hết rồi mới apply 1 lần ("best-improvement per step").
        // Lý do cần duyệt hết: vì có tabu list, move tốt của đỉnh đầu
        // có thể bị cấm, còn move tốt nhất thực sự nằm ở đỉnh sau.
        // ══════════════════════════════════════════════════════════════════
        int    bestImprU = -1, bestImprFrom = -1, bestImprTo = -1;
        double bestImprScore = -SCORE_EPS; // Chỉ cập nhật khi score < -SCORE_EPS (thực sự cải thiện)

        int    fallbackU = -1, fallbackFrom = -1, fallbackTo = -1;
        double fallbackScore = 1e300; // Khởi tạo rất lớn, cập nhật move ít xấu nhất

        // Duyệt từng đỉnh trong nodeOrder
        for (int u : nodeOrder)
        {
            int from = sol.assign[u]; // Cluster hiện tại của u
            // Không di chuyển u nếu cluster 'from' chỉ còn 1 thành viên
            // (tránh cluster rỗng sau khi u rời đi)
            if ((int)sol.members[from].size() <= 1) continue;

            // Kiểm tra tabu: tabu[u][from] > absStep(step) → u bị cấm rời 'from'
            // (vì move trước đó đã set tabu để cấm reverse move)
            bool uTabuFrom = (tabu[u][from] > absStep(step));

            // Tính trước "phần âm" của deltaDist: -clusterSumDist[u][from]
            // deltaDist = clusterSumDist[u][to] - clusterSumDist[u][from]
            double distBase = -sol.clusterSumDist[u][from];

            // Xét tất cả cluster đích 'to' có thể (K cluster, trừ 'from')
            for (int to = 0; to < K; ++to)
            {
                if (to == from) continue; // Không di chuyển sang chính cluster hiện tại

                // deltaDist: thay đổi intra-distance khi di chuyển u từ 'from' sang 'to'
                // = dist(u đến thành viên 'to') - dist(u đến thành viên 'from')
                // Dùng distBase đã tính trước để tránh tính lại mỗi vòng
                double deltaDist = sol.clusterSumDist[u][to] + distBase;

                // Tính vi phạm SAU KHI di chuyển u: from→to
                double vfA = 0.0, vtA = 0.0;
                for (int t = 0; t < M_weights; ++t) {
                    double sf = sol.clusterWeight[from][t] - Wmat[u][t]; // Trọng số 'from' sau khi mất u
                    double st = sol.clusterWeight[to][t]   + Wmat[u][t]; // Trọng số 'to' sau khi nhận u
                    // Vi phạm của 'from' sau move
                    if      (sf < WLmat[from][t]) vfA += WLmat[from][t] - sf; // Dưới lower bound
                    else if (sf > WUmat[from][t]) vfA += sf - WUmat[from][t]; // Vượt upper bound
                    // Vi phạm của 'to' sau move
                    if      (st < WLmat[to][t])   vtA += WLmat[to][t]   - st;
                    else if (st > WUmat[to][t])   vtA += st - WUmat[to][t];
                }
                // deltaViol: thay đổi vi phạm nếu thực hiện move này
                // = (vi phạm mới của from + to) - (vi phạm hiện tại của from + to)
                double dViol = (vfA + vtA) - (vc[from] + vc[to]);

                // score: hàm mục tiêu kết hợp (intra_dist + penalty × violation)
                // score < 0: move cải thiện hàm mục tiêu
                double score = deltaDist + pen * dViol;

                // Kiểm tra aspiration criterion:
                // Nếu move này bị tabu NHƯNG dẫn đến nghiệm feasible TỐT HƠN bestFeas
                // → override tabu (cho phép thực hiện dù bị cấm)
                bool isTabu = uTabuFrom; // u bị cấm rời 'from'
                if (isTabu) {
                    double newViol  = totalViol + dViol;   // Tổng vi phạm sau move
                    double newIntra = curIntra  + deltaDist; // Intra-dist sau move
                    if (newViol < VALID_EPS && newIntra < aspireFeas - EPS)
                        isTabu = false; // Aspiration override: cho phép move tabu này
                }
                if (isTabu) continue; // Bị cấm và không đủ điều kiện aspiration → bỏ qua

                // Cập nhật bestImpr: move cải thiện tốt nhất (score < -SCORE_EPS)
                if (score < bestImprScore) {
                    bestImprScore = score;
                    bestImprU     = u;
                    bestImprFrom  = from;
                    bestImprTo    = to;
                }

                // Cập nhật fallback: move ít xấu nhất trong tất cả move không bị cấm
                // (dùng khi không có move cải thiện nào)
                if (score < fallbackScore) {
                    fallbackScore = score;
                    fallbackU     = u;
                    fallbackFrom  = from;
                    fallbackTo    = to;
                }
            }
        }

        // ── Lambda: Apply một RELOCATE move ──
        // Gộp vào lambda để tái sử dụng cho cả bestImpr và fallback
        bool didRelocate = false;

        auto applyRelocateMove = [&](int u, int from, int to) {
            // Tính lại deltaDist và dViol chính xác (trước khi apply)
            double dDist = sol.clusterSumDist[u][to] - sol.clusterSumDist[u][from];

            double vfA = 0.0, vtA = 0.0;
            for (int t = 0; t < M_weights; ++t) {
                double sf = sol.clusterWeight[from][t] - Wmat[u][t];
                double st = sol.clusterWeight[to][t]   + Wmat[u][t];
                if      (sf < WLmat[from][t]) vfA += WLmat[from][t] - sf;
                else if (sf > WUmat[from][t]) vfA += sf - WUmat[from][t];
                if      (st < WLmat[to][t])   vtA += WLmat[to][t]   - st;
                else if (st > WUmat[to][t])   vtA += st - WUmat[to][t];
            }
            double dViol = (vfA + vtA) - (vc[from] + vc[to]);

            // Thực hiện di chuyển: cập nhật toàn bộ cấu trúc dữ liệu
            ts_relocate(sol, u, from, to);
            vc[from]  = computeClViol(from); // Tính lại vi phạm của 'from' sau move
            vc[to]    = computeClViol(to);   // Tính lại vi phạm của 'to' sau move
            totalViol += dViol; totalViol = max(totalViol, 0.0); // Cập nhật tổng vi phạm
            curIntra  += dDist;                                   // Cập nhật intra-distance

            // Set tabu: cấm u rời cluster 'from' trong 'tenure' bước tiếp theo
            // Tenure ngẫu nhiên: tabuBase ± tabuDelta (tránh hiện tượng chu kỳ)
            int tenure = tabuBase + tenureRand(rng);
            tabu[u][from] = absStep(step) + max(1, tenure); // Đảm bảo tenure ≥ 1

            updatePen(); // Cập nhật penalty dựa trên trạng thái feasibility sau move

            // Cập nhật bestFeas nếu nghiệm sau move là feasible và tốt hơn
            if (totalViol < VALID_EPS && (!hasBestFeas || curIntra < bestFeasIntra - EPS)) {
                hasBestFeas   = true;
                bestFeasIntra = curIntra;
                bestFeasSol   = sol;
                aspireFeas    = bestFeasIntra; // Cập nhật ngưỡng aspiration
            }
        };

        // Apply move cải thiện nếu tìm được
        if (bestImprU >= 0) {
            applyRelocateMove(bestImprU, bestImprFrom, bestImprTo);
            didRelocate = true; // Đánh dấu đã apply relocate thành công
        }

        if (didRelocate) {
            continue; // Đã tìm và apply move cải thiện → sang bước tiếp theo
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE B — SWAP: first-improvement với CL và random
        //
        // Chỉ chạy khi Phase A không tìm được move cải thiện.
        // Dùng first-improvement (apply ngay khi tìm thấy) để nhanh hơn.
        // Duyệt toàn fullOrder (không phải focusedNodes) vì swap cần xét
        // cặp đỉnh từ 2 cluster khác nhau, không giới hạn trong focused set.
        // ══════════════════════════════════════════════════════════════════
        bool foundSwap = false;

        // Lambda thử thực hiện swap giữa u và v
        // Trả về true nếu swap được thực hiện (tìm thấy cải thiện và không bị tabu)
        auto trySwap = [&](int u, int v) -> bool {
            if (u == v) return false;           // Không swap với chính mình
            int cu = sol.assign[u], cv = sol.assign[v]; // Cluster của u và v
            if (cu == cv) return false;          // Cùng cluster → swap vô nghĩa

            // Tính deltaDist khi hoán đổi u↔v:
            // u đến cv: sol.clusterSumDist[u][cv] - sol.clusterSumDist[u][cu]
            // v đến cu: sol.clusterSumDist[v][cu] - sol.clusterSumDist[v][cv]
            // Trừ đi dist(u,v) × 2 vì u và v không còn trong cùng cluster sau swap
            // (Nhưng thực ra u và v ở 2 cluster khác nhau cả trước và sau → trừ 2× vì cộng đôi)
            double deltaDist =
                (sol.clusterSumDist[u][cv] - sol.clusterSumDist[u][cu])
              + (sol.clusterSumDist[v][cu] - sol.clusterSumDist[v][cv])
              - distmat[u][v] - distmat[v][u]; // Trừ vì dist(u,v) bị đếm 2 lần ở trên

            // Tính vi phạm sau swap: cu nhận v mất u, cv nhận u mất v
            double vcuA = 0.0, vcvA = 0.0;
            for (int t = 0; t < M_weights; ++t) {
                double scu = sol.clusterWeight[cu][t] - Wmat[u][t] + Wmat[v][t]; // cu: -u +v
                double scv = sol.clusterWeight[cv][t] - Wmat[v][t] + Wmat[u][t]; // cv: -v +u
                if      (scu < WLmat[cu][t]) vcuA += WLmat[cu][t] - scu;
                else if (scu > WUmat[cu][t]) vcuA += scu - WUmat[cu][t];
                if      (scv < WLmat[cv][t]) vcvA += WLmat[cv][t] - scv;
                else if (scv > WUmat[cv][t]) vcvA += scv - WUmat[cv][t];
            }
            double dViol = (vcuA + vcvA) - (vc[cu] + vc[cv]); // deltaViol của swap
            double score = deltaDist + pen * dViol;

            // Kiểm tra tabu cho swap: cả u lẫn v đều không được bị cấm rời cluster hiện tại
            bool isTabu = (tabu[u][cu] > absStep(step)) || (tabu[v][cv] > absStep(step));
            if (isTabu) {
                // Aspiration: override nếu nghiệm mới feasible + tốt hơn bestFeas
                double newViol  = totalViol + dViol;
                double newIntra = curIntra  + deltaDist;
                if (newViol < VALID_EPS && newIntra < aspireFeas - EPS)
                    isTabu = false;
            }
            if (isTabu) return false;  // Bị cấm và không thỏa aspiration → bỏ qua
            if (score >= -SCORE_EPS) return false; // Không cải thiện → bỏ qua (SWAP chỉ chấp nhận cải thiện)

            // Apply swap
            ts_swap(sol, u, v);
            vc[cu]     = computeClViol(cu);
            vc[cv]     = computeClViol(cv);
            totalViol += dViol; totalViol = max(totalViol, 0.0);
            curIntra  += deltaDist;

            // Set tabu cho cả 2 đỉnh sau swap: cấm u rời cu, cấm v rời cv
            int tenure = tabuBase + tenureRand(rng);
            tabu[u][cu] = absStep(step) + max(1, tenure);
            tabu[v][cv] = absStep(step) + max(1, tenure);

            updatePen();

            if (totalViol < VALID_EPS && (!hasBestFeas || curIntra < bestFeasIntra - EPS)) {
                hasBestFeas   = true;
                bestFeasIntra = curIntra;
                bestFeasSol   = sol;
                aspireFeas    = bestFeasIntra;
            }
            return true; // Swap đã được thực hiện thành công
        };

        // B1: Swap với neighbor trong Candidate List (những đỉnh gần nhất)
        // CL chứa các đỉnh gần nhau về mặt không gian → swap giữa chúng có khả năng
        // cải thiện intra-dist cao hơn so với swap ngẫu nhiên.
        for (int u : fullOrder) {
            if (foundSwap) break;          // First-improvement: dừng ngay khi tìm được
            for (int v : cl[u]) {          // v là neighbor gần của u trong CL
                if (trySwap(u, v)) { foundSwap = true; break; }
            }
        }

        // B2: Random swap — khám phá lân cận ngoài CL
        // Xét cặp đỉnh ngẫu nhiên để thoát khỏi vùng bị kẹt
        if (!foundSwap) {
            const int RAND_TRIES = min(N, 30); // Tối đa 30 lần thử ngẫu nhiên
            for (int r = 0; r < RAND_TRIES && !foundSwap; ++r) {
                int u = randNodeDist(rng), v = randNodeDist(rng); // 2 đỉnh ngẫu nhiên
                if (trySwap(u, v)) foundSwap = true;
            }
        }

        if (foundSwap) {
            continue; // Đã apply swap cải thiện → sang bước tiếp theo
        }

        // ══════════════════════════════════════════════════════════════════
        // FALLBACK: Apply move ít xấu nhất
        //
        // Không tìm được move cải thiện ở cả RELOCATE lẫn SWAP.
        // → Đây là lúc TS thể hiện sức mạnh: chấp nhận trạng thái xấu hơn
        //   để thoát khỏi local optimum.
        //
        // Gọi updatePenIdle TRƯỚC khi apply:
        //   - Feasible + kẹt → giảm pen ngay (mở cửa sổ, dễ cải thiện hơn)
        //   - Infeasible + kẹt → tăng pen ngay (ép mạnh về feasible)
        // ══════════════════════════════════════════════════════════════════
        updatePenIdle(); // Đổi penalty ngay lập tức

        if (fallbackU >= 0) {
            applyRelocateMove(fallbackU, fallbackFrom, fallbackTo); // Apply move ít xấu nhất
        } else {
            // Tất cả move đều bị tabu, không có aspiration override nào khả dụng
            break; // Dừng vòng lặp TS (mọi move đều bị cấm)
        }
    } // end for step
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 3: PERTURBATION — Cluster Segment Transfer
//
// Cơ chế phá vỡ nghiệm (diversification) để thoát khỏi local optimum.
//
// CÁCH HOẠT ĐỘNG:
//   1. Chọn ngẫu nhiên 'pertSize' đỉnh từ N đỉnh.
//   2. Với mỗi đỉnh u, tính xác suất di chuyển đến mỗi cluster 'to' ≠ cluster hiện tại.
//      Xác suất tỷ lệ nghịch với clusterSumDist[u][to] (cluster gần hơn → prob cao hơn).
//      Thêm hệ số smoothing (maxD × 0.1) để cluster xa cũng có cơ hội được chọn.
//   3. Dùng roulette wheel selection để chọn cluster đích.
//   4. Sau khi di chuyển u: set tabu[u][oldK] với tenure dài (×2) để bảo vệ
//      perturbation khỏi bị TS focused hoàn tác ngay lập tức.
//
// TRẢ VỀ: danh sách đỉnh đã thực sự bị di chuyển (để build focusedSet cho TS focused).
// ═══════════════════════════════════════════════════════════════════════════

static vector<int> perturbSolution(
    ACOSolution         &sol,       // Nghiệm bị perturbation trực tiếp
    mt19937_64          &rng,       // Bộ sinh số ngẫu nhiên
    int                  pertSize,  // Số đỉnh muốn di chuyển
    vector<vector<int>> &tabu,      // Tabu list shared (cập nhật tabu sau perturbation)
    int                  stepOffset,// Bước tuyệt đối hiện tại
    int                  tabuBase)  // Tenure cơ bản (tenure perturbation = tabuBase × 2)
{
    if (pertSize <= 0) return {}; // Không perturbation gì

    uniform_real_distribution<double> uni(0.0, 1.0); // Phân phối đều [0,1) cho roulette wheel

    // Tạo hoán vị ngẫu nhiên của {0,...,N-1} và lấy pertSize đỉnh đầu tiên
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0); // {0, 1, ..., N-1}
    shuffle(indices.begin(), indices.end(), rng);
    indices.resize(min(pertSize, N)); // Lấy tối đa min(pertSize, N) đỉnh

    vector<int> moved;      // Danh sách đỉnh thực sự bị di chuyển
    moved.reserve(pertSize);

    for (int u : indices)
    {
        int oldK = sol.assign[u]; // Cluster hiện tại của u

        // Không di chuyển nếu cluster 'oldK' chỉ còn 1 thành viên (tránh cluster rỗng)
        if ((int)sol.members[oldK].size() <= 1) continue;

        // Tính xác suất cho mỗi cluster đích:
        // prob[k] = maxD - clusterSumDist[u][k] + maxD × 0.1
        // → Cluster gần u hơn (clusterSumDist nhỏ) → prob cao hơn.
        // → maxD × 0.1 là smoothing để cluster xa vẫn có xác suất dương.
        double maxD = *max_element(sol.clusterSumDist[u].begin(),
                                   sol.clusterSumDist[u].end()); // Khoảng cách lớn nhất
        maxD = max(maxD, 1.0); // Đảm bảo maxD > 0 (tránh chia cho 0 hoặc prob âm)

        vector<double> prob(K, 0.0);
        double sumP = 0.0;
        for (int k = 0; k < K; ++k) {
            if (k == oldK) continue; // Không di chuyển đến chính cluster hiện tại
            prob[k] = (maxD - sol.clusterSumDist[u][k]) + maxD * 0.1; // Tính xác suất
            if (prob[k] < 0.0) prob[k] = 0.0; // Đảm bảo không âm
            sumP += prob[k];
        }
        if (sumP < 1e-12) continue; // Tổng xác suất quá nhỏ → bỏ qua đỉnh này

        // Roulette wheel selection: chọn cluster với xác suất tỷ lệ với prob[k]
        double r = uni(rng) * sumP; // Điểm ngẫu nhiên trên tổng xác suất
        double acc = 0.0;
        int newK = oldK; // Giá trị mặc định (không thay đổi)
        for (int k = 0; k < K; ++k) {
            acc += prob[k]; // Tích lũy xác suất
            if (r <= acc) { newK = k; break; } // Chọn cluster k khi vượt điểm r
        }
        if (newK == oldK) continue; // Không chọn được cluster mới → bỏ qua

        // Thực hiện di chuyển u từ oldK sang newK
        ts_relocate(sol, u, oldK, newK);

        // Khóa reverse move với tenure dài gấp đôi:
        // tabu[u][oldK] = stepOffset + tabuBase × 2
        // → Bảo vệ perturbation: TS focused không thể ngay lập tức hoàn tác move này.
        tabu[u][oldK] = stepOffset + tabuBase * 2;

        moved.push_back(u); // Ghi nhận u đã bị di chuyển
    }

    // Cập nhật cost và feasibility sau khi perturbation hoàn tất
    sol.cost     = compute_cost_fast(sol);
    sol.feasible = is_feasible(sol.assign);
    return moved; // Trả danh sách đỉnh đã di chuyển (để xây dựng focusedSet)
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 4: BUILD FOCUSED SET
//
// Xây dựng tập hợp đỉnh cần tìm kiếm sau perturbation.
// Bao gồm: các đỉnh bị perturbation + các neighbor của chúng trong CL.
//
// Lý do: perturbation ảnh hưởng trực tiếp đến các đỉnh bị di chuyển
// và gián tiếp đến các neighbor gần của chúng (thay đổi cluster membership
// có thể tạo cơ hội di chuyển tốt hơn cho neighbor).
//
// Dùng vector<bool> để tránh thêm duplicate (O(N) không gian, O(1) kiểm tra).
// ═══════════════════════════════════════════════════════════════════════════

static vector<int> buildFocusedSet(
    const vector<int>         &perturbed, // Danh sách đỉnh đã bị perturbation
    const vector<vector<int>> &cl)        // Candidate List
{
    vector<bool> inSet(N, false); // Mảng đánh dấu: inSet[i] = true → i thuộc focusedSet

    // Đánh dấu tất cả đỉnh bị perturbation
    for (int u : perturbed) inSet[u] = true;

    // Đánh dấu tất cả neighbor trong CL của các đỉnh bị perturbation
    for (int u : perturbed)
        for (int v : cl[u]) inSet[v] = true;

    // Thu thập thành vector để dùng trong ts_core (focusedNodes)
    vector<int> result;
    for (int i = 0; i < N; ++i)
        if (inSet[i]) result.push_back(i);
    return result;
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 5: PUBLIC API — iterated_tabu_search
//
// Hàm chính được gọi từ bên ngoài. Nhận nghiệm đầu vào 'sol' (thường là
// kết quả của Local Search) và cải thiện nó bằng Iterated Tabu Search.
//
// LUỒNG CHI TIẾT:
//   0. Khởi tạo: xây dựng CL, tabu list, tracking variables.
//   1. TS warm-up: khai thác sâu nghiệm đầu vào.
//   2. Vòng lặp ILS (maxRounds vòng):
//      a. Perturbation → khóa tabu
//      b. Build focusedSet
//      c. TS focused (focusedSet, focusedSteps bước)
//      d. TS full-pass (toàn N, fullSteps bước)
//      e. Acceptance với cửa sổ chấp nhận (thu hẹp dần khi stagnate)
//      f. Restart khi tắc nghẽn quá lâu
//   3. Trả về bestFeas (nếu có) hoặc bestOverall.
//
// BUDGET STEPS:
//   Tổng ≈ tsSteps (warm-up) + maxRounds × (focusedSteps + fullSteps)
//   focusedSteps = tsSteps / 2, fullSteps = tsSteps / 2 (chia đôi mỗi round)
// ═══════════════════════════════════════════════════════════════════════════

void iterated_tabu_search(ACOSolution &sol, mt19937_64 &rng)
{
    if (N <= 0 || K <= 0) return; // Kiểm tra đầu vào hợp lệ

    const double EPS = 1e-9; // Ngưỡng so sánh số thực

    // ── Các tham số thuật toán ──
    // tsSteps: số bước cho mỗi lần gọi ts_core.
    //   Được điều chỉnh theo N: bài toán lớn hơn → cần nhiều bước hơn.
    //   Giới hạn trong [50, 800] để cân bằng chất lượng và thời gian.
    const int tsSteps = max(50, min(N * 3, 800));

    // maxRounds: số vòng lặp ILS.
    //   1000 / tsSteps: tổng budget ~ 1000 bước ngoài warm-up.
    //   Giới hạn trong [5, 20].
    const int maxRounds = max(5, min(20, 1000 / max(tsSteps, 1)));

    // focusedSteps và fullSteps: phân chia budget mỗi round.
    const int focusedSteps = tsSteps / 2;         // TS focused: nửa budget
    const int fullSteps    = tsSteps - focusedSteps; // TS full-pass: nửa còn lại

    // pertSize: số đỉnh bị perturbation mỗi round.
    //   Tối thiểu K+1 đỉnh (để có thể phá ít nhất 1 đỉnh/cluster bình quân).
    //   Tối thiểu 3 để có ý nghĩa.
    //   Tối đa N/4 (không phá quá nhiều một lúc).
    const int pertSize = max(max(K + 1, 3), min(N / 4, N / 8 + K));

    // noImproveRestart: số vòng không cải thiện trước khi restart.
    const int noImproveRestart = max(2, maxRounds / 3);

    // tabuBase và tabuDelta: kiểm soát độ dài tenure của tabu list.
    //   tenure ∈ [tabuBase - tabuDelta, tabuBase + tabuDelta].
    //   Randomization tránh hiện tượng chu kỳ trong TS.
    const int tabuBase  = max(4, min(N / 12, 12)); // Tenure cơ bản: tỷ lệ với N, tối đa 12
    const int tabuDelta = max(2, tabuBase / 3);     // Biên độ ngẫu nhiên: 1/3 của tabuBase

    // ── Xây dựng Candidate List (CL) ──
    // cl[i] = danh sách CL_SIZE đỉnh gần i nhất (theo khoảng cách trung bình 2 chiều).
    // CL được dùng trong SWAP phase của ts_core để ưu tiên xét các neighbor gần.
    const int CL_SIZE = min(20, N - 1); // Tối đa 20 neighbor, tối thiểu N-1
    vector<vector<int>> cl(N);

    // Kiểm tra nếu globalCL đã được build sẵn (từ ACO hoặc lần gọi trước)
    if (!globalCL.empty() && (int)globalCL.size() == N) {
        cl = globalCL; // Dùng lại CL đã có → tránh tính lại O(N²)
    } else {
        // Build CL từ đầu: với mỗi đỉnh i, tìm CL_SIZE đỉnh j gần nhất
        vector<pair<double, int>> tmp(N); // {khoảng_cách, chỉ_số_đỉnh}
        for (int i = 0; i < N; ++i) {
            // Tính khoảng cách trung bình 2 chiều: (dist(i,j) + dist(j,i)) / 2
            // Dùng khoảng cách đối xứng để CL ổn định hơn khi distmat không đối xứng
            for (int j = 0; j < N; ++j)
                tmp[j] = {(distmat[i][j] + distmat[j][i]) * 0.5, j};
            tmp[i].first = 1e300; // Loại i ra khỏi danh sách neighbor của chính nó
            // partial_sort: chỉ sắp xếp CL_SIZE phần tử nhỏ nhất, O(N log CL_SIZE)
            partial_sort(tmp.begin(), tmp.begin() + CL_SIZE, tmp.end());
            cl[i].resize(CL_SIZE);
            for (int r = 0; r < CL_SIZE; ++r) cl[i][r] = tmp[r].second; // Lưu chỉ số đỉnh
        }
    }

    // ── Tabu list shared xuyên suốt toàn bộ ITS ──
    // tabu[u][k] = bước tuyệt đối X: u không được rời cluster k đến bước X.
    // Không reset giữa các round ILS (tabu cũ vẫn còn hiệu lực, giúp tránh lặp).
    // Chỉ reset khi stagnation restart (cần khám phá không gian hoàn toàn mới).
    vector<vector<int>> tabu(N, vector<int>(K, 0)); // Khởi tạo tất cả = 0 (không cấm gì)
    int globalStep = 0; // Tổng số bước đã chạy qua tất cả lần gọi ts_core

    // Hàm reset tabu list về trạng thái ban đầu
    auto resetTabu = [&]() {
        for (auto &row : tabu) fill(row.begin(), row.end(), 0); // Đặt tất cả = 0
        globalStep = 0; // Reset bộ đếm bước tuyệt đối
    };

    // ── Tracking: theo dõi nghiệm tốt nhất ──
    ACOSolution bestFeasSol;            // Nghiệm feasible tốt nhất tìm được
    bestFeasSol.feasible = false;
    bestFeasSol.cost     = 1e300;
    double bestFeasIntra  = 1e300;      // Intra-dist của bestFeasSol
    bool   hasBestFeas    = false;      // Flag: đã có nghiệm feasible chưa

    ACOSolution bestOverallSol  = sol;  // Nghiệm tốt nhất tổng thể (kể cả infeasible)
    double      bestOverallCost = sol.cost; // Cost của bestOverallSol

    // computeIntra: tính intra-distance của một nghiệm (không dùng clusterSumDist)
    auto computeIntra = [&](const ACOSolution &s) -> double {
        double d = 0.0;
        for (int i = 0; i < N; ++i)
            d += s.clusterSumDist[i][s.assign[i]]; // Tổng dist(i → members cùng cluster)
        return d;
    };

    // tryUpdate: cập nhật bestFeas và bestOverall nếu nghiệm s tốt hơn
    auto tryUpdate = [&](ACOSolution &s) {
        if (s.feasible) {
            double intra = computeIntra(s);
            if (!hasBestFeas || intra < bestFeasIntra - EPS) { // Cải thiện so với best feasible
                hasBestFeas   = true;
                bestFeasIntra = intra;
                bestFeasSol   = s;
            }
        }
        if (s.cost < bestOverallCost - EPS) { // Cải thiện so với best overall
            bestOverallCost = s.cost;
            bestOverallSol  = s;
        }
    };

    // Khởi tạo tracking từ nghiệm đầu vào
    if (sol.feasible) {
        hasBestFeas    = true;
        bestFeasSol    = sol;
        bestFeasIntra  = computeIntra(sol);
        bestOverallCost = sol.cost;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BƯỚC 1: TS WARM-UP
    //
    // Khai thác sâu hơn local opt của nghiệm đầu vào trước khi vào ILS.
    // Dùng toàn bộ N đỉnh (focusedNodes = nullptr) với tsSteps bước.
    // Sau warm-up: nếu tìm được nghiệm feasible tốt hơn, dùng nó làm điểm khởi đầu ILS.
    // ═══════════════════════════════════════════════════════════════════════
    ACOSolution current = sol; // Nghiệm đang làm việc của ILS
    ts_core(current, rng, tsSteps,
            bestFeasSol, bestFeasIntra, hasBestFeas,
            tabuBase, tabuDelta, cl,
            tabu, globalStep,
            nullptr); // nullptr = duyệt toàn N đỉnh (không focused)
    globalStep += tsSteps; // Cập nhật tổng số bước đã chạy

    current.cost     = compute_cost_fast(current); // Tính lại cost sau TS
    current.feasible = is_feasible(current.assign); // Kiểm tra feasibility
    tryUpdate(current);

    // Bắt đầu ILS từ nghiệm feasible tốt nhất nếu có
    if (hasBestFeas) current = bestFeasSol;

    int    noImprove = 0;     // Số vòng liên tiếp không cải thiện bestFeas
    double window    = 0.02;  // Cửa sổ chấp nhận: tối đa 2% tệ hơn bestFeas
                              // → Cho phép "side step" để thoát local opt

    // ═══════════════════════════════════════════════════════════════════════
    // BƯỚC 2: VÒNG LẶP ILS (Iterated Local Search)
    // ═══════════════════════════════════════════════════════════════════════
    for (int round = 0; round < maxRounds; ++round)
    {
        ACOSolution candidate = current; // Bắt đầu từ bản sao của current

        // ── a. Perturbation + khóa tabu ngay ──
        // Di chuyển pertSize đỉnh ngẫu nhiên, khóa reverse move với tenure dài.
        // 'perturbed' chứa danh sách đỉnh thực sự bị di chuyển (có thể < pertSize
        // nếu một số đỉnh là singleton nên không thể di chuyển).
        vector<int> perturbed = perturbSolution(
            candidate, rng, pertSize, tabu, globalStep, tabuBase);

        if (perturbed.empty()) {
            // Không perturbation được (tất cả đỉnh muốn di chuyển đều singleton)
            ++noImprove; // Tính là 1 vòng không cải thiện
            continue;
        }

        // ── b. Build focusedSet: vùng bị ảnh hưởng bởi perturbation ──
        // = perturbed + CL-neighbors của các đỉnh bị perturbation
        vector<int> focusedSet = buildFocusedSet(perturbed, cl);

        // ── c. TS focused: khai thác vùng vừa bị phá ──
        // Chỉ duyệt focusedSet trong RELOCATE phase (hiệu quả hơn toàn N đỉnh).
        // SWAP phase trong ts_core vẫn duyệt toàn N đỉnh.
        // Kế thừa tabu list từ warm-up (không reset giữa các round).
        ts_core(candidate, rng, focusedSteps,
                bestFeasSol, bestFeasIntra, hasBestFeas,
                tabuBase, tabuDelta, cl,
                tabu, globalStep,
                &focusedSet); // Truyền focusedSet để giới hạn RELOCATE
        globalStep += focusedSteps;

        // ── d. TS full-pass: dọn hiệu ứng lan rộng ──
        // Sau focused TS, có thể vẫn còn cơ hội cải thiện ở các đỉnh khác ngoài focusedSet.
        // Full-pass duyệt toàn N đỉnh để xử lý triệt để.
        ts_core(candidate, rng, fullSteps,
                bestFeasSol, bestFeasIntra, hasBestFeas,
                tabuBase, tabuDelta, cl,
                tabu, globalStep,
                nullptr); // nullptr = toàn N đỉnh
        globalStep += fullSteps;

        candidate.cost     = compute_cost_fast(candidate);
        candidate.feasible = is_feasible(candidate.assign);
        tryUpdate(candidate); // Cập nhật bestFeas và bestOverall nếu candidate tốt hơn

        // ── e. Acceptance criterion với cửa sổ ──
        // Quyết định có cập nhật 'current' = 'candidate' hay không.
        //
        // Điều kiện chấp nhận:
        //   (1) Greedy: candidate tốt hơn current → chấp nhận, reset noImprove
        //   (2) Window: candidate trong khoảng window so với bestFeas → chấp nhận (diversification)
        //   (3) Else: từ chối → tăng noImprove
        bool accept = false;

        if (hasBestFeas && candidate.feasible) {
            // Cả current và candidate đều feasible → so sánh intra-dist
            double candIntra = computeIntra(candidate);
            double curIntra  = computeIntra(current);

            if (candIntra < curIntra - EPS) {
                // Candidate tốt hơn current → greedy accept
                accept    = true;
                noImprove = 0; // Reset đếm vì đã cải thiện
            } else if (candIntra <= bestFeasIntra * (1.0 + window) - EPS) {
                // Candidate trong cửa sổ window so với bestFeas → accept để diversify
                accept = true;
                ++noImprove; // Không cải thiện bestFeas, nhưng vẫn accept
            } else {
                // Candidate quá tệ → từ chối
                ++noImprove;
            }
        } else if (!hasBestFeas) {
            // Chưa có nghiệm feasible nào → so sánh cost tổng thể
            if (candidate.cost < bestOverallCost * (1.0 + window))
                accept = true;
            ++noImprove;
        } else {
            // candidate infeasible nhưng current feasible → từ chối
            ++noImprove;
        }

        if (accept) current = candidate; // Cập nhật current nếu được chấp nhận

        // Thu hẹp cửa sổ dần dần khi không cải thiện (càng lâu không cải thiện, càng conservative)
        window = max(0.001, window * 0.9); // Giảm 10% mỗi vòng, tối thiểu 0.1%

        // ── f. Restart khi tắc nghẽn ──
        // Nếu không cải thiện noImproveRestart vòng liên tiếp → đã mắc kẹt nghiêm trọng.
        // Chiến lược: quay lại nghiệm tốt nhất, reset tabu, perturbation mạnh hơn.
        if (noImprove >= noImproveRestart)
        {
            // Quay lại nghiệm tốt nhất đã biết (best feasible nếu có, ngược lại best overall)
            current   = (hasBestFeas ? bestFeasSol : bestOverallSol);
            noImprove = 0;    // Reset đếm stagnation
            window    = 0.02; // Reset cửa sổ về giá trị ban đầu

            // Reset hoàn toàn tabu list:
            // Các entry cũ không còn ý nghĩa khi đã khám phá không gian mới.
            // Reset cho phép tự do di chuyển bất kỳ đâu.
            resetTabu();

            // Perturbation mạnh hơn thông thường: gấp đôi pertSize
            // Mục đích: nhảy xa hơn khỏi vùng hiện tại, khám phá không gian mới
            int strongPert = min(pertSize * 2, N / 2); // Tối đa N/2 đỉnh
            vector<int> dummy = perturbSolution(
                current, rng, strongPert, tabu, globalStep, tabuBase);
            // Advance globalStep để tenure tabu của perturbation có hiệu lực ngay
            // (tránh ts_core ngay lập tức xóa perturbation)
            globalStep += tabuBase * 2;

            current.cost     = compute_cost_fast(current);
            current.feasible = is_feasible(current.assign);
            tryUpdate(current);
        }
    } // end for round (ILS)

    // ── Trả về nghiệm tốt nhất ──
    // Ưu tiên nghiệm feasible tốt nhất; nếu không có thì dùng best overall.
    sol          = (hasBestFeas ? bestFeasSol : bestOverallSol);
    sol.cost     = compute_cost_fast(sol);     // Tính lại cost chính xác
    sol.feasible = is_feasible(sol.assign);    // Kiểm tra lại feasibility
}
