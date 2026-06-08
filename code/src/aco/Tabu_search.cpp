// ═══════════════════════════════════════════════════════════════════════════
// FILE: Tabu_search.cpp
//
// THUẬT TOÁN: Iterated Tabu Search (ITS) — Tìm kiếm Tabu Lặp
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
//      b. TS focused: TS bình thường nhưng chỉ tìm kiếm trong vùng bị perturbation,
//         KHÔNG dùng fallback — dừng ngay khi bị kẹt (nhanh hơn).
//      c. TS full-pass: tìm kiếm toàn bộ N đỉnh để xử lý hiệu ứng lan rộng,
//         dùng fallback như bình thường.
//      d. Acceptance: chấp nhận nghiệm mới nếu đủ tốt (có cửa sổ chấp nhận).
//      e. Restart: nếu bị tắc nghẽn quá lâu, quay lại nghiệm tốt nhất + perturbation mạnh.
//   3. Trả về bestFeas (nếu có) hoặc bestOverall.
//
// HAI LOẠI MOVE:
//   - RELOCATE: di chuyển 1 đỉnh u từ cluster 'from' sang cluster 'to'.
//   - SWAP: hoán đổi 2 đỉnh u, v đang ở 2 cluster khác nhau.
//
// TABU LIST — CỤC BỘ TỪNG VÒNG CORE, KHÔNG SHARED, KHÔNG PRESET:
//
//   Mỗi lần gọi ts_core khởi tạo tabuList list riêng từ 0 (All Zeros).
//   tabuList[u][k] = bước X: đỉnh u không được ĐẾN cluster k cho đến bước X.
//   "Bước X" được tính theo step cục bộ trong chính vòng core đó (0-indexed).
//
//   Sau khi di chuyển u: from→to tại bước 'step':
//     tenure = tabuBase + rand(-tabuDelta, +tabuDelta)
//     tabuList[u][from] = step + max(1, tenure)
//     → Hiệu lực: lệnh cấm hết hạn vào bước 'step + tenure' của vòng core này.
//     → Vì tenure << maxSteps thông thường, lệnh cấm thường hết trong chính vòng đó.
//
//   CHECK khi xét move u→to tại bước 'step':
//     if (tabuList[u][to] > step) → move bị cấm.
//
//   LƯU Ý THIẾT KẾ MỚI:
//     Caller (iterated_tabu_search) KHÔNG set trước (preset) bất kỳ giá trị cấm nào
//     trước khi gọi ts_core. Dù là Warm-up, Focused hay Full-pass, tabuList luôn
//     được truyền vào dưới dạng toàn số 0. Ts_core sẽ tự tạo ra lịch sử cấm
//     của riêng nó trong quá trình chạy dựa trên các move nó thực hiện.
//
// CƠ CHẾ HOẠT ĐỘNG TRONG 1 BƯỚC TS CORE:
//   1. PHASE ĐÁNH GIÁ (EVALUATION):
//      - Đánh giá TẤT CẢ các move Relocate có thể. Lưu lại 2 trạng thái:
//        (A) Move cải thiện tốt nhất (nếu có).
//        (B) Move fallback ít xấu nhất (nếu tất cả đều xấu).
//      - Nếu có (A), BỎ QUA việc đánh giá Swap.
//      - Nếu KHÔNG có (A), tiến hành đánh giá TẤT CẢ các move Swap. Lưu lại:
//        (C) Swap cải thiện tốt nhất (nếu có).
//        (D) Swap fallback ít xấu nhất (nếu tất cả đều xấu).
//   2. PHASE THỰC THI (EXECUTION):
//      - Ưu tiên 1: Nếu có (A) -> Thực thi Relocate (A).
//      - Ưu tiên 2: Nếu có (C) -> Thực thi Swap (C).
//      - Xử lý kẹt: Nếu allowFallback=true -> So sánh (B) và (D), thực thi cái ít xấu hơn.
//      - Dừng sớm: Nếu allowFallback=false -> Dừng vòng lặp ngay.
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

// ts_erase: xóa phần tử 'node' khỏi vector 'v' bằng cách swap với phần tử
//           cuối rồi pop_back — thao tác O(N) find + O(1) xóa.
static inline void ts_erase(vector<int> &v, int node) {
    // Tìm vị trí của 'node' trong vector
    auto it = find(v.begin(), v.end(), node);
    if (it != v.end()) {        // Nếu tìm thấy
        *it = v.back();         // Ghi đè bằng phần tử cuối (tránh dịch chuyển mảng)
        v.pop_back();           // Xóa phần tử cuối (giờ là bản sao của 'node')
    }
}

// ts_relocate: di chuyển đỉnh u từ cluster 'from' sang cluster 'to',
//              đồng thời cập nhật incremental toàn bộ cấu trúc nghiệm.
static void ts_relocate(ACOSolution &sol, int u, int from, int to)
{
    sol.assign[u] = to;              // Cập nhật nhãn cluster của u thành 'to'
    ts_erase(sol.members[from], u);  // Xóa u khỏi danh sách thành viên của cluster 'from'
    sol.members[to].push_back(u);    // Thêm u vào danh sách thành viên của cluster 'to'

    // Cập nhật trọng số của cluster 'from' và 'to' sau khi u di chuyển
    for (int t = 0; t < M_weights; ++t) {
        sol.clusterWeight[from][t] -= Wmat[u][t]; // Trừ trọng số u ra khỏi cluster 'from'
        sol.clusterWeight[to][t]   += Wmat[u][t]; // Cộng trọng số u vào cluster 'to'
    }

    // Cập nhật clusterSumDist incremental:
    // Với mọi đỉnh v != u, khoảng cách từ v đến cluster 'from' giảm đi distmat[v][u],
    // và khoảng cách từ v đến cluster 'to' tăng thêm distmat[v][u].
    for (int v = 0; v < N; ++v) {
        if (v == u) continue;      // Bỏ qua chính đỉnh u
        double d = distmat[v][u];  // Khoảng cách từ v đến u
        sol.clusterSumDist[v][from] -= d; // v mất đi khoảng cách đến u trong 'from'
        sol.clusterSumDist[v][to]   += d; // v có thêm khoảng cách đến u trong 'to'
    }
}

// ts_swap: hoán đổi vị trí 2 đỉnh u và v đang ở 2 cluster khác nhau (cu ≠ cv),
//          cập nhật incremental toàn bộ cấu trúc nghiệm.
static void ts_swap(ACOSolution &sol, int u, int v)
{
    int cu = sol.assign[u]; // Cluster hiện tại của u
    int cv = sol.assign[v]; // Cluster hiện tại của v
    if (cu == cv) return;   // Không swap nếu cùng cluster (vô nghĩa)

    sol.assign[u] = cv; sol.assign[v] = cu; // Đổi nhãn cluster: u sang cv, v sang cu
    ts_erase(sol.members[cu], u); ts_erase(sol.members[cv], v); // Xóa u khỏi cu, v khỏi cv
    sol.members[cv].push_back(u); sol.members[cu].push_back(v); // Thêm u vào cv, v vào cu

    // Cập nhật trọng số cluster: cu mất u nhưng nhận v, cv mất v nhưng nhận u
    for (int t = 0; t < M_weights; ++t) {
        sol.clusterWeight[cu][t] += Wmat[v][t] - Wmat[u][t]; // cu: +v -u
        sol.clusterWeight[cv][t] += Wmat[u][t] - Wmat[v][t]; // cv: +u -v
    }

    // Cập nhật clusterSumDist cho tất cả đỉnh w != u, v:
    // w đóng góp khoảng cách vào cluster của từng thành viên.
    // Sau swap: w cần thêm d(w,v) vào cu (vì v giờ thuộc cu) và bớt d(w,u) (u rời cu).
    //           Tương tự với cv.
    for (int w = 0; w < N; ++w) {
        if (w == u || w == v) continue; // Bỏ qua chính u và v — xử lý riêng bên dưới
        double du = distmat[w][u]; // Khoảng cách từ w đến u
        double dv = distmat[w][v]; // Khoảng cách từ w đến v
        sol.clusterSumDist[w][cu] += dv - du; // cu: mất u (+du→−du), nhận v (thêm dv)
        sol.clusterSumDist[w][cv] += du - dv; // cv: mất v (+dv→−dv), nhận u (thêm du)
    }

    // Xử lý riêng cho cặp (u, v) vì chúng là 2 đỉnh vừa hoán đổi:
    // u giờ thuộc cv → tổng khoảng cách của u đến cv tăng thêm d(u,v) (u gặp v trong cv)
    //                  và tổng khoảng cách của u đến cu giảm đi d(u,v) (u rời cu, không còn gặp v)
    double duv = distmat[u][v]; // Khoảng cách u→v
    double dvu = distmat[v][u]; // Khoảng cách v→u (có thể khác nếu ma trận không đối xứng)
    sol.clusterSumDist[u][cu] += duv; // u rời cu → cu mất u nhưng v vào cu: d(u,v) được thêm
    sol.clusterSumDist[u][cv] -= duv; // u vào cv → u không còn cách biệt v: trừ d(u,v)
    sol.clusterSumDist[v][cv] += dvu; // v rời cv → cv mất v nhưng u vào cv: d(v,u) được thêm
    sol.clusterSumDist[v][cu] -= dvu; // v vào cu → v không còn cách biệt u: trừ d(v,u)
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 2: TS CORE — Một phiên Tìm kiếm Tabu
//
// Đánh giá toàn diện Relocate và Swap, sau đó mới quyết định thực thi.
// ═══════════════════════════════════════════════════════════════════════════

static void ts_core(
    ACOSolution               &sol,          // Nghiệm hiện tại (được sửa đổi trực tiếp)
    mt19937_64                &rng,          // Bộ sinh số ngẫu nhiên (64-bit Mersenne Twister)
    int                        maxSteps,     // Số bước tối đa của vòng lặp TS
    ACOSolution               &bestFeasSol,  // Nghiệm feasible tốt nhất tìm được (output)
    double                    &bestFeasIntra,// Tổng intra-distance của bestFeasSol
    bool                      &hasBestFeas, // True nếu đã tìm được ít nhất 1 nghiệm feasible
    int                        tabuBase,    // Giá trị cơ bản của tenure (độ dài cấm)
    int                        tabuDelta,   // Biên độ ngẫu nhiên của tenure (±tabuDelta)
    const vector<vector<int>> &cl,          // Candidate List: cl[u] = danh sách đỉnh gần u nhất
    vector<vector<int>>       &tabuList,    // Danh sách cấm: tabuList[u][k] = bước kết thúc cấm
    const vector<int>         *focusedNodes = nullptr, // Nếu khác nullptr: chỉ xét tập đỉnh này
    bool                       allowFallback = true)   // Cho phép thực hiện move xấu hơn không?
{
    const double SCORE_EPS = 1e-9; // Ngưỡng cải thiện tối thiểu để coi là "tốt hơn"

    // Lambda tính tổng vi phạm ràng buộc trọng số của cluster k.
    // Vi phạm = tổng phần vượt quá upper bound + tổng phần thiếu dưới lower bound.
    auto computeClViol = [&](int k) -> double {
        double v = 0.0;
        for (int t = 0; t < M_weights; ++t) {
            double s = sol.clusterWeight[k][t]; // Trọng số chiều t của cluster k
            if (s > WUmat[k][t]) v += s - WUmat[k][t]; // Vượt upper bound → cộng phần thừa
            if (s < WLmat[k][t]) v += WLmat[k][t] - s; // Dưới lower bound → cộng phần thiếu
        }
        return v; // Tổng vi phạm của cluster k
    };

    // Khởi tạo mảng vi phạm cho từng cluster và tổng vi phạm toàn cục
    vector<double> vc(K); // vc[k] = vi phạm của cluster k
    double totalViol = 0.0;
    for (int k = 0; k < K; ++k) {
        vc[k]      = computeClViol(k); // Tính vi phạm cluster k
        totalViol += vc[k];            // Cộng dồn tổng vi phạm
    }
    totalViol = max(totalViol, 0.0); // Đảm bảo không âm do lỗi làm tròn

    // Tính tổng intra-cluster distance hiện tại:
    // intra-distance = tổng khoảng cách từ mỗi đỉnh i đến tất cả đỉnh cùng cluster
    double curIntra = 0.0;
    for (int i = 0; i < N; ++i)
        curIntra += sol.clusterSumDist[i][sol.assign[i]]; // Cộng dồn khoảng cách nội cluster của i

    // Nếu nghiệm hiện tại feasible (totalViol ≈ 0) và tốt hơn bestFeas → cập nhật
    if (totalViol < VALID_EPS) {
        if (!hasBestFeas || curIntra < bestFeasIntra - VALID_EPS) {
            hasBestFeas   = true;     // Đã có nghiệm feasible
            bestFeasIntra = curIntra; // Cập nhật intra tốt nhất
            bestFeasSol   = sol;      // Lưu nghiệm feasible tốt nhất
        }
    }

    // ── Khởi tạo Dynamic Penalty ──
    double pen = PENALTY_SCALE;                    // Giá trị ban đầu của hệ số phạt
    const double PEN_UP     = 2.0;                 // Hệ số tăng penalty khi liên tục infeasible
    const double PEN_DOWN   = 2.0;                 // Hệ số giảm penalty khi feasible + kẹt
    const double PEN_MIN    = PENALTY_SCALE * 0.2; // Giới hạn dưới của penalty
    const double PEN_MAX    = PENALTY_SCALE * 5.0; // Giới hạn trên của penalty
    const int    gradSteps  = min(10, max(N / 20, 10));                  // Số bước infeasible liên tiếp trước khi tăng pen
    int stepsInfeas = 0;                           // Đếm số bước liên tiếp ở trạng thái infeasible

    // Lambda updatePen: gọi sau mỗi bước có move được thực thi.
    // Nếu infeasible: tăng đếm, khi đủ gradSteps → tăng pen để phạt nặng hơn.
    // Nếu feasible: reset đếm (không tăng pen).
    auto updatePen = [&]() {
        if (totalViol < VALID_EPS) {
            stepsInfeas = 0; // Feasible → reset đếm infeasible
        } else {
            ++stepsInfeas; // Infeasible → tăng đếm
            if (stepsInfeas >= gradSteps) {
                pen         = min(pen * PEN_UP, PEN_MAX); // Tăng penalty, không vượt PEN_MAX
                stepsInfeas = 0;                          // Reset đếm sau khi điều chỉnh
            }
        }
    };

    // Lambda updatePenIdle: gọi khi bị kẹt (không có move cải thiện, dùng fallback).
    // Khi feasible + kẹt → giảm penalty để mở rộng không gian tìm kiếm.
    // Khi infeasible + kẹt → tăng penalty để đẩy về feasible.
    auto updatePenIdle = [&]() {
        if (totalViol < VALID_EPS) {
            pen         = max(pen / PEN_DOWN, PEN_MIN); // Feasible + kẹt → giảm penalty
            stepsInfeas = 0;
        } else {
            pen         = min(pen * PEN_UP, PEN_MAX); // Infeasible + kẹt → tăng penalty
            stepsInfeas = 0;
        }
    };

    // Phân phối ngẫu nhiên uniform trong [-tabuDelta, +tabuDelta] cho tenure
    uniform_int_distribution<int> tenureRand(-tabuDelta, tabuDelta);
    // Phân phối ngẫu nhiên chọn đỉnh bất kỳ trong [0, N-1] cho swap ngẫu nhiên
    uniform_int_distribution<int> randNodeDist(0, N - 1);

    // Xác định tập đỉnh cần duyệt:
    // Nếu focusedNodes được cung cấp → dùng tập đó (focused mode).
    // Nếu không → duyệt toàn bộ N đỉnh.
    vector<int> nodeOrder;
    if (focusedNodes && !focusedNodes->empty()) {
        nodeOrder = *focusedNodes; // Chế độ focused: chỉ duyệt tập đỉnh được chỉ định
    } else {
        nodeOrder.resize(N);
        iota(nodeOrder.begin(), nodeOrder.end(), 0); // Chế độ full: duyệt tất cả đỉnh 0..N-1
    }
    shuffle(nodeOrder.begin(), nodeOrder.end(), rng); // Xáo ngẫu nhiên thứ tự duyệt

    // aspireFeas: ngưỡng aspiration — nếu move tabu nhưng cho nghiệm feasible tốt hơn
    // ngưỡng này → vẫn cho phép thực hiện (override tabu).
    double aspireFeas = hasBestFeas ? bestFeasIntra : 1e300;

    // ══════════════════════════════════════════════════════════════════════
    // VÒNG LẶP CHÍNH TS CORE
    // Mỗi bước: đánh giá tất cả move hợp lệ → chọn và thực thi move tốt nhất.
    // ══════════════════════════════════════════════════════════════════════
    for (int step = 0; step < maxSteps; ++step)
    {
        // Xoay vòng nodeOrder: đưa phần tử đầu lên cuối để đảm bảo
        // mỗi bước bắt đầu từ đỉnh khác nhau (round-robin cộng với shuffle ban đầu).
        rotate(nodeOrder.begin(), nodeOrder.begin() + 1, nodeOrder.end());

        // ══════════════════════════════════════════════════════════════════
        // KHAI BÁO BIẾN LƯU TRỮ KẾT QUẢ ĐÁNH GIÁ
        // Mỗi bước cần tìm: (A) relocate cải thiện tốt nhất, (B) relocate fallback,
        //                   (C) swap cải thiện tốt nhất, (D) swap fallback.
        // ══════════════════════════════════════════════════════════════════

        // --- BIẾN CHO RELOCATE ---
        int    bestImprU = -1, bestImprFrom = -1, bestImprTo = -1; // Đỉnh và cluster của move cải thiện tốt nhất
        double bestImprScore = -SCORE_EPS; // Điểm tốt nhất của relocate cải thiện (âm = tốt hơn)
        double bestImprDDist = 0.0, bestImprDViol = 0.0; // Delta khoảng cách và vi phạm của move đó

        int    fallbackU = -1, fallbackFrom = -1, fallbackTo = -1; // Đỉnh và cluster của relocate fallback
        double fallbackScore = 1e300; // Điểm fallback (nhỏ nhất trong các move xấu)
        double fallbackDDist = 0.0, fallbackDViol = 0.0; // Delta của move fallback

        // --- BIẾN CHO SWAP ---
        int    bestSwapU = -1, bestSwapV = -1; // Cặp đỉnh của swap cải thiện tốt nhất
        double bestSwapScore = -SCORE_EPS; // Điểm tốt nhất của swap cải thiện
        double bestSwapDDist = 0.0, bestSwapDViol = 0.0; // Delta của swap đó

        int    fallbackSwapU = -1, fallbackSwapV = -1; // Cặp đỉnh của swap fallback
        double fallbackSwapScore = 1e300; // Điểm fallback của swap
        double fallbackSwapDDist = 0.0, fallbackSwapDViol = 0.0; // Delta của swap fallback

        // ══════════════════════════════════════════════════════════════════
        // PHASE A — ĐÁNH GIÁ TẤT CẢ RELOCATE (CHỈ ĐÁNH GIÁ, CHƯA THỰC THI)
        // Với mỗi đỉnh u trong nodeOrder, xét tất cả cluster to ≠ from.
        // Tính delta intra-distance và delta vi phạm nếu u chuyển sang to.
        // ══════════════════════════════════════════════════════════════════
        for (int u : nodeOrder)
        {
            int from = sol.assign[u]; // Cluster hiện tại của u
            // Bỏ qua nếu cluster 'from' chỉ có 1 thành viên — không được lấy đi u
            if ((int)sol.members[from].size() <= 1) continue;

            // distBase: phần đóng góp khoảng cách của u tới cluster 'from' (âm vì sẽ bị xóa)
            double distBase = -sol.clusterSumDist[u][from];

            // Xét tất cả cluster 'to' mà u có thể chuyển sang
            for (int to = 0; to < K; ++to)
            {
                if (to == from) continue; // Không xét chuyển về chính cluster hiện tại

                // deltaDist: thay đổi intra-distance nếu u rời 'from' và vào 'to'
                // = (tổng kc từ u đến các đỉnh trong 'to') - (tổng kc từ u đến các đỉnh trong 'from')
                double deltaDist = sol.clusterSumDist[u][to] + distBase;

                // Tính vi phạm của cluster 'from' sau khi mất u (vfA) và 'to' sau khi nhận u (vtA)
                double vfA = 0.0, vtA = 0.0;
                for (int t = 0; t < M_weights; ++t) {
                    double sf = sol.clusterWeight[from][t] - Wmat[u][t]; // Trọng số 'from' sau khi mất u
                    double st = sol.clusterWeight[to][t]   + Wmat[u][t]; // Trọng số 'to' sau khi nhận u
                    // Vi phạm của 'from' sau move
                    if      (sf < WLmat[from][t]) vfA += WLmat[from][t] - sf; // Thiếu lower bound
                    else if (sf > WUmat[from][t]) vfA += sf - WUmat[from][t]; // Vượt upper bound
                    // Vi phạm của 'to' sau move
                    if      (st < WLmat[to][t])   vtA += WLmat[to][t]   - st; // Thiếu lower bound
                    else if (st > WUmat[to][t])   vtA += st - WUmat[to][t];   // Vượt upper bound
                }
                // dViol: thay đổi vi phạm so với hiện tại của 2 cluster liên quan
                double dViol = (vfA + vtA) - (vc[from] + vc[to]);
                // score: hàm mục tiêu có phạt = delta_dist + pen * delta_vi_phạm
                double score = deltaDist + pen * dViol;

                // Kiểm tra xem move u→to có bị cấm bởi tabuList không
                bool isTabu = (tabuList[u][to] > step); // Cấm nếu thời gian cấm chưa hết

                // Aspiration criterion: override tabu nếu move cho nghiệm feasible tốt hơn bestFeas
                if (isTabu) {
                    double newViol  = totalViol + dViol;  // Vi phạm mới sau move
                    double newIntra = curIntra  + deltaDist; // Intra mới sau move
                    // Nếu feasible VÀ tốt hơn aspireFeas → cho phép dù bị tabu
                    if (newViol < VALID_EPS && newIntra < aspireFeas - VALID_EPS)
                        isTabu = false; // Vượt qua aspiration → hủy cấm
                }
                if (isTabu) continue; // Vẫn bị cấm → bỏ qua move này

                // Cập nhật move cải thiện tốt nhất: score < 0 nghĩa là cải thiện thực sự
                if (score < bestImprScore) {
                    bestImprScore = score;  // Cập nhật điểm tốt nhất
                    bestImprU     = u; bestImprFrom  = from; bestImprTo = to; // Lưu thông tin move
                    bestImprDDist = deltaDist; bestImprDViol = dViol; // Lưu delta để áp dụng
                }
                // Cập nhật fallback: move ít xấu nhất trong tất cả move hợp lệ
                if (score < fallbackScore) {
                    fallbackScore = score;   // Cập nhật điểm fallback tốt nhất
                    fallbackU     = u; fallbackFrom  = from; fallbackTo = to; // Lưu thông tin
                    fallbackDDist = deltaDist; fallbackDViol = dViol; // Lưu delta
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE B — ĐÁNH GIÁ SWAP (CHỈ CHẠY NẾU RELOCATE KHÔNG TÌM ĐƯỢC CẢI THIỆN)
        // Nếu đã có relocate cải thiện (bestImprU >= 0) → bỏ qua swap để tiết kiệm thời gian.
        // ══════════════════════════════════════════════════════════════════
        if (bestImprU < 0) // Chỉ đánh giá Swap nếu Relocate không tìm được cải thiện
        {
            // Lambda nội bộ để đánh giá một cặp swap (u, v)
            auto evaluateSwap = [&](int u, int v) {
                if (u == v) return; // Không swap đỉnh với chính nó
                int cu = sol.assign[u], cv = sol.assign[v]; // Cluster của u và v
                if (cu == cv) return; // Không swap 2 đỉnh cùng cluster (vô nghĩa)

                // deltaDist: thay đổi intra-distance nếu u và v hoán đổi cluster
                // Công thức: (kc u→cv - kc u→cu) + (kc v→cu - kc v→cv) - d(u,v) - d(v,u)
                // (trừ d(u,v) và d(v,u) vì u và v không còn ở cùng cluster sau swap)
                double deltaDist =
                    (sol.clusterSumDist[u][cv] - sol.clusterSumDist[u][cu])
                  + (sol.clusterSumDist[v][cu] - sol.clusterSumDist[v][cv])
                  - distmat[u][v] - distmat[v][u]; // Trừ khoảng cách u↔v (đã tính 2 chiều)

                // Tính vi phạm của cu và cv sau khi hoán đổi
                double vcuA = 0.0, vcvA = 0.0;
                for (int t = 0; t < M_weights; ++t) {
                    double scu = sol.clusterWeight[cu][t] - Wmat[u][t] + Wmat[v][t]; // cu: mất u, nhận v
                    double scv = sol.clusterWeight[cv][t] - Wmat[v][t] + Wmat[u][t]; // cv: mất v, nhận u
                    // Vi phạm cu sau swap
                    if      (scu < WLmat[cu][t]) vcuA += WLmat[cu][t] - scu;
                    else if (scu > WUmat[cu][t]) vcuA += scu - WUmat[cu][t];
                    // Vi phạm cv sau swap
                    if      (scv < WLmat[cv][t]) vcvA += WLmat[cv][t] - scv;
                    else if (scv > WUmat[cv][t]) vcvA += scv - WUmat[cv][t];
                }
                // dViol: thay đổi vi phạm của 2 cluster sau swap
                double dViol = (vcuA + vcvA) - (vc[cu] + vc[cv]);
                // score = delta_dist + pen * delta_vi_phạm
                double score = deltaDist + pen * dViol;

                // Swap bị cấm nếu tabu chặn u→cv HOẶC v→cu
                bool isTabu = (tabuList[u][cv] > step) || (tabuList[v][cu] > step);

                // Aspiration criterion: override nếu cho nghiệm feasible tốt hơn
                if (isTabu) {
                    double newViol  = totalViol + dViol;
                    double newIntra = curIntra  + deltaDist;
                    if (newViol < VALID_EPS && newIntra < aspireFeas - VALID_EPS)
                        isTabu = false; // Cho phép vượt tabu nếu đạt aspiration
                }
                if (isTabu) return; // Vẫn bị cấm → bỏ qua swap này

                // Cập nhật swap cải thiện tốt nhất
                if (score < bestSwapScore) {
                    bestSwapScore = score;
                    bestSwapU = u; bestSwapV = v;
                    bestSwapDDist = deltaDist; bestSwapDViol = dViol;
                }
                // Cập nhật fallback swap
                if (score < fallbackSwapScore) {
                    fallbackSwapScore = score;
                    fallbackSwapU = u; fallbackSwapV = v;
                    fallbackSwapDDist = deltaDist; fallbackSwapDViol = dViol;
                }
            };

            // Đánh giá swap trong Candidate List:
            // cl[u] chứa các đỉnh gần u nhất → swap giữa u và hàng xóm gần thường hiệu quả hơn
            for (int u : nodeOrder) {
                for (int v : cl[u]) {
                    evaluateSwap(u, v); // Đánh giá swap u với từng đỉnh trong CL của u
                }
            }

            // Đánh giá thêm N × Rand_tries cặp swap ngẫu nhiên để đa dạng hóa tìm kiếm:
            // Duyệt từng đỉnh u trong nodeOrder (N đỉnh), mỗi đỉnh thử swap với
            // Rand_tries đỉnh v ngẫu nhiên → tổng N × Rand_tries lần thử.
            const int Rand_tries = 10;
            for (int u : nodeOrder) {
                for (int r = 0; r < Rand_tries; ++r) {
                    int v = randNodeDist(rng); // Chọn đỉnh v ngẫu nhiên trong [0, N-1]
                    evaluateSwap(u, v);
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE C — THỰC THI MOVE (EXECUTION)
        // Ưu tiên: (1) Relocate cải thiện → (2) Swap cải thiện → (3) Fallback/Dừng
        // ══════════════════════════════════════════════════════════════════

        // --- LAMBDA THỰC THI RELOCATE ---
        // Áp dụng move relocate, cập nhật các biến tracking, đặt tabu.
        auto applyRelocateMove = [&](int u, int from, int to, double dDist, double dViol) {
            ts_relocate(sol, u, from, to); // Thực hiện di chuyển u từ 'from' sang 'to'
            vc[from]   = computeClViol(from); // Tính lại vi phạm của cluster 'from'
            vc[to]     = computeClViol(to);   // Tính lại vi phạm của cluster 'to'
            totalViol += dViol; totalViol = max(totalViol, 0.0); // Cập nhật tổng vi phạm
            curIntra  += dDist; // Cập nhật tổng intra-distance

            // Đặt tabu: sau khi u rời 'from', cấm u quay lại 'from' trong 'tenure' bước tới
            int tenure = tabuBase + tenureRand(rng); // Tenure ngẫu nhiên quanh tabuBase
            tabuList[u][from] = step + max(1, tenure); // Cấm u→from đến bước step+tenure

            updatePen(); // Điều chỉnh dynamic penalty dựa trên trạng thái feasible hiện tại

            // Cập nhật bestFeasSol nếu nghiệm hiện tại feasible và tốt hơn tốt nhất đã biết
            if (totalViol < VALID_EPS && (!hasBestFeas || curIntra < bestFeasIntra - VALID_EPS)) {
                hasBestFeas   = true;
                bestFeasIntra = curIntra;
                bestFeasSol   = sol;
                aspireFeas    = bestFeasIntra; // Cập nhật ngưỡng aspiration
            }
        };

        // --- LAMBDA THỰC THI SWAP ---
        // Áp dụng swap, cập nhật tracking, đặt tabu cho cả u và v.
        auto applySwapMove = [&](int u, int v, double dDist, double dViol) {
            int cu = sol.assign[u], cv = sol.assign[v]; // Cluster cũ của u và v trước khi swap

            ts_swap(sol, u, v); // Thực hiện hoán đổi u và v
            vc[cu]     = computeClViol(cu); // Tính lại vi phạm cu
            vc[cv]     = computeClViol(cv); // Tính lại vi phạm cv
            totalViol += dViol; totalViol = max(totalViol, 0.0); // Cập nhật tổng vi phạm
            curIntra  += dDist; // Cập nhật tổng intra-distance

            // Đặt tabu cho cả 2 đỉnh: u không quay về cu, v không quay về cv
            int tenure = tabuBase + tenureRand(rng);    // Tenure ngẫu nhiên
            tabuList[u][cu] = step + max(1, tenure);    // Cấm u→cu trong tenure bước
            tabuList[v][cv] = step + max(1, tenure);    // Cấm v→cv trong tenure bước

            updatePen(); // Điều chỉnh penalty

            // Cập nhật bestFeasSol nếu tốt hơn
            if (totalViol < VALID_EPS && (!hasBestFeas || curIntra < bestFeasIntra - VALID_EPS)) {
                hasBestFeas   = true;
                bestFeasIntra = curIntra;
                bestFeasSol   = sol;
                aspireFeas    = bestFeasIntra;
            }
        };

        // --- LOGIC CHỌN MOVE ĐỂ THỰC THI ---

        // Ưu tiên 1: Nếu có Relocate cải thiện → thực thi ngay, bỏ qua phần còn lại
        if (bestImprU >= 0) {
            applyRelocateMove(bestImprU, bestImprFrom, bestImprTo, bestImprDDist, bestImprDViol);
            continue; // Sang bước tiếp theo của vòng lặp step
        }

        // Ưu tiên 2: Nếu có Swap cải thiện → thực thi
        if (bestSwapU >= 0) {
            applySwapMove(bestSwapU, bestSwapV, bestSwapDDist, bestSwapDViol);
            continue; // Sang bước tiếp theo
        }

        // Không có move nào cải thiện → xử lý tình trạng kẹt
        if (!allowFallback) {
            break; // Chế độ focused (allowFallback=false): dừng ngay khi kẹt
        }

        updatePenIdle(); // Điều chỉnh penalty vì đang kẹt (không có move cải thiện)

        // So sánh Fallback Relocate và Fallback Swap → chọn move ít xấu nhất
        bool canRelocateFallback = (fallbackU >= 0);       // Có fallback relocate không?
        bool canSwapFallback    = (fallbackSwapU >= 0);    // Có fallback swap không?

        if (canRelocateFallback && canSwapFallback) {
            // Cả hai đều có → chọn cái có score nhỏ hơn (ít xấu hơn)
            if (fallbackScore <= fallbackSwapScore) {
                applyRelocateMove(fallbackU, fallbackFrom, fallbackTo, fallbackDDist, fallbackDViol);
            } else {
                applySwapMove(fallbackSwapU, fallbackSwapV, fallbackSwapDDist, fallbackSwapDViol);
            }
        }
        else if (canRelocateFallback) {
            applyRelocateMove(fallbackU, fallbackFrom, fallbackTo, fallbackDDist, fallbackDViol);
        }
        else if (canSwapFallback) {
            applySwapMove(fallbackSwapU, fallbackSwapV, fallbackSwapDDist, fallbackSwapDViol);
        }
        else {
            break; // Mọi move đều bị Tabu chặn hoàn toàn → thoát vòng lặp
        }
    } // end for step — kết thúc 1 phiên ts_core
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 3: PERTURBATION — Cluster Segment Transfer
//
// Di chuyển ngẫu nhiên một số đỉnh sang cluster khác để "phá vỡ" cấu trúc
// nghiệm hiện tại, giúp ILS thoát khỏi local optimum và khám phá vùng mới.
// ═══════════════════════════════════════════════════════════════════════════

static vector<int> perturbSolution(
    ACOSolution         &sol,       // Nghiệm bị perturbation (sửa trực tiếp)
    mt19937_64          &rng,       // Bộ sinh số ngẫu nhiên
    int                  pertSize)  // Số đỉnh muốn perturb
{
    if (pertSize <= 0) return {}; // Không perturb nếu pertSize = 0

    uniform_real_distribution<double> uni(0.0, 1.0); // Phân phối đều [0,1] để roulette wheel

    // Tạo và xáo ngẫu nhiên danh sách đỉnh để chọn đỉnh perturb ngẫu nhiên
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0); // indices = {0, 1, 2, ..., N-1}
    shuffle(indices.begin(), indices.end(), rng); // Xáo ngẫu nhiên
    indices.resize(min(pertSize, N)); // Giữ lại tối đa pertSize đỉnh

    vector<int> moved; // Danh sách các đỉnh thực sự bị di chuyển
    moved.reserve(pertSize); // Đặt sẵn bộ nhớ để tránh realloc

    for (int u : indices)
    {
        int oldK = sol.assign[u]; // Cluster hiện tại của u
        // Bỏ qua nếu cluster đó chỉ có 1 thành viên → không được lấy u đi
        if ((int)sol.members[oldK].size() <= 1) continue;

        // Tìm giá trị clusterSumDist lớn nhất của u để chuẩn hóa xác suất
        double maxD = *max_element(sol.clusterSumDist[u].begin(),
                                   sol.clusterSumDist[u].end());
        maxD = max(maxD, 1.0); // Tránh chia cho 0

        // Tính xác suất chọn cluster đích theo roulette wheel:
        // Cluster nào có clusterSumDist nhỏ hơn (gần u hơn) → xác suất cao hơn.
        // prob[k] = (maxD - clusterSumDist[u][k]) + maxD*0.1 (thêm baseline để tránh prob=0)
        vector<double> prob(K, 0.0);
        double sumP = 0.0;
        for (int k = 0; k < K; ++k) {
            if (k == oldK) continue; // Không chọn cluster hiện tại
            prob[k] = (maxD - sol.clusterSumDist[u][k]) + maxD * 0.1; // Xác suất tỉ lệ nghịch với khoảng cách
            if (prob[k] < 0.0) prob[k] = 0.0; // Đảm bảo không âm
            sumP += prob[k]; // Tích lũy tổng xác suất để chuẩn hóa
        }
        if (sumP < 1e-12) continue; // Nếu tổng xác suất quá nhỏ → bỏ qua đỉnh này

        // Chọn cluster đích bằng roulette wheel selection
        double r = uni(rng) * sumP; // Giá trị ngẫu nhiên trong [0, sumP]
        double acc = 0.0;
        int newK = oldK; // Mặc định giữ nguyên (nếu vòng lặp không chọn được)
        for (int k = 0; k < K; ++k) {
            acc += prob[k]; // Tích lũy dần
            if (r <= acc) { newK = k; break; } // Dừng khi tích lũy vượt r → chọn k
        }
        if (newK == oldK) continue; // Roulette chọn về cluster cũ → bỏ qua

        ts_relocate(sol, u, oldK, newK); // Di chuyển u sang cluster mới
        moved.push_back(u);              // Ghi nhận u đã bị perturb
    }

    // Cập nhật cost và tình trạng feasible sau perturbation
    sol.cost     = compute_cost_fast(sol);
    sol.feasible = is_feasible(sol.assign);
    return moved; // Trả về danh sách đỉnh đã bị perturb
}


// ═══════════════════════════════════════════════════════════════════════════
// PHẦN 4: PUBLIC API — iterated_tabu_search
//
// Hàm chính được gọi từ bên ngoài.
//
// THIẾT KẾ TABU: KHÔNG SHARED, MỖI VÒNG CORE TỰ RESET
//
//   Thay vì dùng 1 tabu list chung xuyên toàn bộ ITS (như thiết kế cũ),
//   mỗi lần gọi ts_core nhận một tabu list đã được RESET VỀ 0 hoặc PRESET
//   trước khi gọi. Lý do:
//
//   - Tabu được set trong ts_core dùng step CỤC BỘ (0-indexed trong vòng đó).
//   - Tenure = tabuBase + rand(±tabuDelta), thường << maxSteps.
//   - Lệnh cấm hết hạn trong chính vòng core đó; giá trị thừa trong tabu
//     list sau khi ts_core kết thúc không có ý nghĩa gì cho vòng tiếp theo.
//   - Vì vậy: RESET tabu về 0 trước mỗi lần gọi ts_core là đúng đắn.
//
//   NGOẠI LỆ (PRESET): tabu cho đỉnh perturbed được set TRƯỚC KHI gọi ts_core
//   với giá trị đặc biệt (không phải 0) để bảo vệ perturbation:
//
//   (A) Trước TS FOCUSED:
//     tabu[u][oldK] = focusedSteps
//     → CHECK trong focused: tabu[u][oldK] > step với step ∈ [0, focusedSteps-1]
//       → focusedSteps > step → luôn đúng → cấm toàn vòng focused.
//
//   (B) Trước TS FULL-PASS:
//     tenure = rand(tabuBase - tabuDelta, tabuBase + tabuDelta)
//     tabu[u][oldK] = max(1, tenure)
//     → CHECK trong full-pass: tabu[u][oldK] > step với step = 0,1,...
//       → Cấm u đến oldK trong tenure bước đầu của full-pass (bình thường).
//       → Sau tenure bước, tự do.
//
//   Sau warm-up và sau full-pass: tabu = all zeros (reset bởi lambda resetTabu).
//   Trước focused và trước full-pass: tabu = all zeros + preset cho perturbed.
// ═══════════════════════════════════════════════════════════════════════════

void iterated_tabu_search(ACOSolution &sol, mt19937_64 &rng)
{
    if (N <= 0 || K <= 0) return; // Kiểm tra đầu vào hợp lệ

    const double VALID_EPS = 1e-9; // Ngưỡng so sánh số thực (tránh lỗi làm tròn)

    // ── Tham số thuật toán ──
    // tsSteps: số bước mỗi lần gọi ts_core — tối thiểu 50, tối đa 500, mặc định N*3
    const int tsSteps      = max(50, min(N * 3, 500));
    // maxRounds: số vòng lặp ILS (mỗi vòng gồm 1 perturbation + focused + full-pass)
    const int maxRounds    = 10;
    // focusedSteps: số bước tối đa của ts focused (nhỏ hơn tsSteps để nhanh hơn)
    const int focusedSteps = tsSteps / 2;
    // pertSize: số đỉnh perturb mỗi vòng (15% số đỉnh, tối thiểu 1)
    const int pertSize     = max(N * 0.15, 1.0);
    // noImproveRestart: số vòng không cải thiện trước khi thực hiện restart
    const int noImproveRestart = 3;

    // tabuBase, tabuDelta: kiểm soát độ dài tenure.
    // tenure ∈ [tabuBase - tabuDelta, tabuBase + tabuDelta].
    const int tabuBase  = max(4, min(N / 10, 12)); // Tenure cơ bản: 4..12, tỉ lệ N
    const int tabuDelta = max(2, tabuBase / 3);    // Biên độ ngẫu nhiên: tối thiểu 2

    // ── Xây dựng Candidate List (CL) ──
    const vector<vector<int>> *clPtr = nullptr;  // con trỏ tới candidate list sẽ dùng
    vector<vector<int>> localCL_fallback;        // CL dự phòng (chỉ xây nếu cần)

    if (!globalCL.empty()) {
        // globalCL hợp lệ (đã xây sẵn trong ACO_tuned) → dùng trực tiếp
        clPtr = &globalCL;
    } else {
        // globalCL chưa có hoặc kích thước sai → xây CL cục bộ
        const int CL_SIZE_LOCAL = min(20, N - 1); // mỗi node có tối đa 20 ứng viên
        localCL_fallback.resize(N);               // kích thước N × CL_SIZE_LOCAL
        vector<pair<double, int>> tmp(N);          // mảng tạm để sort (dist, node_id)
        for (int i = 0; i < N; ++i) {
            // Tính khoảng cách trung bình từ i đến mọi node j
            for (int j = 0; j < N; ++j)
                tmp[j] = {(distmat[i][j] + distmat[j][i]), j}; // dist(i,j)
            tmp[i].first = 1e300; // loại bỏ node i chính nó (đặt dist = vô cực)
            // Chọn CL_SIZE_LOCAL node gần nhất bằng partial_sort (nhanh hơn sort đầy đủ)
            partial_sort(tmp.begin(), tmp.begin() + CL_SIZE_LOCAL, tmp.end());
            localCL_fallback[i].resize(CL_SIZE_LOCAL);
            for (int r = 0; r < CL_SIZE_LOCAL; ++r)
                localCL_fallback[i][r] = tmp[r].second; // lưu index node (không lưu dist)
        }
        clPtr = &localCL_fallback; // trỏ đến CL vừa xây
    }
    const vector<vector<int>> &cl = *clPtr; // candidate list toàn bộ đỉnh

    // ── Tabu list — cục bộ từng vòng core, KHÔNG shared ──
    //
    // tabuList[u][k] = bước cục bộ khi lệnh cấm u→k kết thúc.
    // Giá trị 0 = không cấm (step luôn bắt đầu từ 0, check là tabuList[u][k] > step).
    vector<vector<int>> tabuList(N, vector<int>(K, 0)); // Khởi tạo toàn 0

    // Lambda resetTabuList: đặt tất cả về 0 — xóa mọi lệnh cấm còn sót từ vòng trước.
    auto resetTabuList = [&]() {
        for (auto &row : tabuList) fill(row.begin(), row.end(), 0);
    };

    // Phân phối tenure dùng khi preset tabu bên ngoài ts_core (cho đỉnh perturbed)
    uniform_int_distribution<int> tenureRandOuter(-tabuDelta, tabuDelta);

    // ── Tracking nghiệm tốt nhất ──
    ACOSolution bestFeasSol;       // Nghiệm feasible tốt nhất từ trước đến nay
    bestFeasSol.feasible = false;
    bestFeasSol.cost     = 1e300;
    double bestFeasIntra  = 1e300; // Intra-distance của bestFeasSol
    bool   hasBestFeas    = false; // Đã có nghiệm feasible chưa?

    ACOSolution bestOverallSol  = sol;      // Nghiệm tốt nhất tổng thể (kể cả infeasible)
    double      bestOverallCost = sol.cost; // Chi phí của bestOverallSol

    // Lambda tính tổng intra-distance của nghiệm s
    auto computeIntra = [&](const ACOSolution &s) -> double {
        double d = 0.0;
        for (int i = 0; i < N; ++i)
            d += s.clusterSumDist[i][s.assign[i]]; // Cộng dồn khoảng cách nội cluster của i
        return d;
    };

    // Lambda tryUpdate: cập nhật bestFeasSol và bestOverallSol nếu s tốt hơn
    auto tryUpdate = [&](ACOSolution &s) {
        if (s.feasible) { // Chỉ cập nhật bestFeas nếu s là feasible
            double intra = computeIntra(s);
            if (!hasBestFeas || intra < bestFeasIntra - VALID_EPS) {
                hasBestFeas   = true;
                bestFeasIntra = intra;
                bestFeasSol   = s; // Lưu nghiệm feasible tốt nhất
            }
        }
        if (s.cost < bestOverallCost - VALID_EPS) {
            bestOverallCost = s.cost;
            bestOverallSol  = s; // Lưu nghiệm tổng thể tốt nhất
        }
    };

    // Nếu nghiệm đầu vào đã feasible → dùng làm điểm khởi đầu bestFeas
    if (sol.feasible) {
        hasBestFeas     = true;
        bestFeasSol     = sol;
        bestFeasIntra   = computeIntra(sol);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BƯỚC 1: TS WARM-UP
    //
    // Khai thác sâu nghiệm đầu vào. Toàn N đỉnh, tsSteps bước, allowFallback = true.
    // Tabu list = all zeros (không có gì bị cấm trước).
    // ═══════════════════════════════════════════════════════════════════════
    ACOSolution current = sol; // current = nghiệm đang làm việc (bắt đầu từ sol)

    // tabuList đã là all zeros → không cần resetTabuList() ở đây
    ts_core(current, rng, tsSteps,
            bestFeasSol, bestFeasIntra, hasBestFeas,
            tabuBase, tabuDelta, cl,
            tabuList,     // tabu = all zeros (không có gì bị cấm trước warm-up)
            nullptr,      // focusedNodes = nullptr → duyệt toàn N đỉnh
            true);        // allowFallback = true → cho phép dùng fallback khi kẹt

    // Cập nhật cost và feasibility sau warm-up
    current.cost     = compute_cost_fast(current);
    current.feasible = is_feasible(current.assign);
    tryUpdate(current); // Kiểm tra và lưu nếu là nghiệm tốt hơn

    int    noImprove = 0;    // Đếm số vòng liên tiếp không cải thiện
    double window    = 0.02; // Cửa sổ chấp nhận ban đầu: 2% tệ hơn bestFeas vẫn được chấp nhận

    // ═══════════════════════════════════════════════════════════════════════
    // BƯỚC 2: VÒNG LẶP ILS
    // Mỗi vòng: perturb → TS focused → TS full-pass → acceptance → (restart nếu cần)
    // ═══════════════════════════════════════════════════════════════════════
    for (int round = 0; round < maxRounds; ++round)
    {
        ACOSolution candidate = current; // Bắt đầu mỗi vòng từ nghiệm hiện tại

        // ── a. Lưu assign trước perturbation ──
        // Cần biết cluster cũ của mỗi đỉnh bị perturb để set tabuList đúng sau đó.
        vector<int> beforeAssign(N);
        for (int i = 0; i < N; ++i) beforeAssign[i] = candidate.assign[i]; // Snapshot assign

        // ── b. Perturbation ──
        // Di chuyển ngẫu nhiên pertSize đỉnh sang cluster khác để thoát local opt.
        vector<int> perturbed = perturbSolution(candidate, rng, pertSize);

        if (perturbed.empty()) {
            // Perturbation thất bại (không có đỉnh nào di chuyển được) → bỏ qua vòng này
            ++noImprove;
            continue;
        }

        // ── c. FOCUSED SET ──
        // focusedSet = tập đỉnh vừa bị perturb → TS focused chỉ tối ưu hóa các đỉnh này
        const vector<int> &focusedSet = perturbed;

        // ── d. RESET TABU + PRESET CHO TS FOCUSED ──
        //
        // Reset tabu về 0 trước (xóa mọi entry cũ từ warm-up).
        // Sau đó preset các đỉnh perturbed: cấm quay về cluster cũ trong TOÀN vòng focused.
        //
        // Cơ chế:
        //   tabuList[u][oldK] = focusedSteps
        //   → CHECK trong focused: tabuList[u][oldK] > step với step ∈ [0, focusedSteps-1]
        //     → focusedSteps > step là luôn đúng (vì step < focusedSteps)
        //     → u KHÔNG THỂ quay về oldK trong suốt vòng focused.
        //
        // Aspiration criterion vẫn có thể override nếu đây là nghiệm feasible tốt nhất.
        resetTabuList(); // Xóa mọi giá trị cũ từ warm-up
        for (int u : perturbed) {
            int oldK = beforeAssign[u]; // Cluster cũ của u trước khi bị perturb
            tabuList[u][oldK] = focusedSteps;
            // → Cấm u đến oldK trong TOÀN BỘ vòng focused.
            // → step cục bộ trong focused: 0 ≤ step < focusedSteps
            // → tabuList[u][oldK] = focusedSteps > step → luôn bị chặn.
        }

        // ── e. TS FOCUSED ──
        // Chỉ tối ưu tập focusedSet. allowFallback=false → dừng ngay khi kẹt.
        // Nhanh hơn vì tập đỉnh nhỏ và dừng sớm.
        ts_core(candidate, rng, focusedSteps,
                bestFeasSol, bestFeasIntra, hasBestFeas,
                tabuBase, tabuDelta, cl,
                tabuList,         // tabu: all zeros + preset perturbed = focusedSteps
                &focusedSet,      // Chỉ duyệt focusedSet
                false);           // allowFallback=false: dừng ngay khi kẹt

        // ── f. RESET TABU + PRESET CHO TS FULL-PASS ──
        //
        // Reset tabu về 0 (xóa entry từ focused — các giá trị đó chỉ có nghĩa trong focused).
        // Preset đỉnh perturbed với tenure BÌNH THƯỜNG tính từ step 0 của full-pass.
        //
        // Cơ chế:
        //   tenure = rand(tabuBase - tabuDelta, tabuBase + tabuDelta)
        //   tabuList[u][oldK] = max(1, tenure)
        //   → CHECK trong full-pass: tabuList[u][oldK] > step
        //     → Đúng khi step < max(1, tenure) — tức tenure bước đầu của full-pass.
        //     → Sau tenure bước, u tự do quay về oldK nếu TS quyết định.
        resetTabuList(); // Xóa entry từ vòng focused
        for (int u : perturbed) {
            int oldK = beforeAssign[u]; // Cluster cũ trước perturbation
            int tenure = tabuBase + tenureRandOuter(rng); // Tenure ngẫu nhiên trong [base±delta]
            tabuList[u][oldK] = max(1, tenure);
            // → Cấm u đến oldK trong tenure bước đầu của full-pass.
            // → Bảo vệ perturbation có thời hạn (không cứng nhắc như focused).
        }

        // ── g. TS FULL-PASS ──
        // Tìm kiếm toàn bộ N đỉnh với fallback. Khai thác sâu hơn sau perturbation.
        ts_core(candidate, rng, tsSteps,
                bestFeasSol, bestFeasIntra, hasBestFeas,
                tabuBase, tabuDelta, cl,
                tabuList,    // tabu: all zeros + preset perturbed = max(1, tenure)
                nullptr,     // focusedNodes = nullptr → toàn N đỉnh
                true);       // allowFallback = true → cho phép fallback

        // Reset tabu sau full-pass để dọn sạch cho vòng ILS tiếp theo
        resetTabuList();

        // Cập nhật cost và feasibility của candidate sau full-pass
        candidate.cost     = compute_cost_fast(candidate);
        candidate.feasible = is_feasible(candidate.assign);
        tryUpdate(candidate); // Cập nhật bestFeas và bestOverall nếu cần

        // ── h. Acceptance criterion ──
        // Quyết định có cập nhật current thành candidate không.
        bool accept = false;

        if (hasBestFeas && candidate.feasible) {
            // Cả current và candidate đều feasible → so sánh theo intra-distance
            double candIntra = computeIntra(candidate);
            double curIntra  = computeIntra(current);

            if (candIntra < curIntra - VALID_EPS) {
                // Candidate cải thiện thực sự → chấp nhận và reset đếm stagnation
                accept    = true;
                noImprove = 0;
            } else if (candIntra <= bestFeasIntra * (1.0 + window) - VALID_EPS) {
                // Candidate trong cửa sổ chấp nhận (tệ hơn bestFeas tối đa 'window'%) → chấp nhận để diversify
                accept = true;
                ++noImprove; // Vẫn tăng đếm stagnation (không phải cải thiện thực sự)
            } else {
                ++noImprove; // Từ chối → tăng đếm stagnation
            }
        } else if (!hasBestFeas) {
            // Chưa có nghiệm feasible → chấp nhận nếu candidate không quá tệ hơn best overall
            if (candidate.cost < bestOverallCost * (1.0 + window))
                accept = true;
            ++noImprove;
        } else {
            // candidate infeasible nhưng current feasible → từ chối (ưu tiên feasibility)
            ++noImprove;
        }

        if (accept) current = candidate; // Cập nhật current nếu được chấp nhận

        // Thu hẹp cửa sổ chấp nhận dần: mỗi vòng giảm 10%, tối thiểu 0.1%
        // → Càng về sau càng ít chấp nhận nghiệm tệ hơn (exploitation > exploration)
        window = max(0.001, window * 0.9);

        // ── i. Restart khi tắc nghẽn (stagnation) ──
        // Nếu noImprove đạt ngưỡng → quay về nghiệm tốt nhất + perturbation mạnh hơn
        if (noImprove >= noImproveRestart)
        {
            // Quay về nghiệm tốt nhất đã biết (feasible ưu tiên, không feasible thì best overall)
            current   = (hasBestFeas ? bestFeasSol : bestOverallSol);
            noImprove = 0;    // Reset đếm stagnation
            window    = 0.02; // Mở rộng lại cửa sổ chấp nhận

            // Perturbation mạnh hơn bình thường (1.5× pertSize, tối đa 25% N)
            for (int i = 0; i < N; ++i) beforeAssign[i] = current.assign[i]; // Lưu assign hiện tại
            int strongPert = min(pertSize * 1.5, N * 0.25); // Tăng số đỉnh perturb
            vector<int> strongPerturbed = perturbSolution(current, rng, strongPert);

            // Cập nhật cost/feasibility sau perturbation mạnh
            current.cost     = compute_cost_fast(current);
            current.feasible = is_feasible(current.assign);
            tryUpdate(current); // Kiểm tra cập nhật best
        }
    } // end for round — kết thúc vòng lặp ILS

    // ── Trả về nghiệm tốt nhất ──
    // Ưu tiên bestFeasSol (feasible), nếu không có thì bestOverallSol
    sol          = (hasBestFeas ? bestFeasSol : bestOverallSol);
    sol.cost     = compute_cost_fast(sol);      // Tính lại cost chính xác
    sol.feasible = is_feasible(sol.assign);     // Xác nhận lại tính feasible
}
