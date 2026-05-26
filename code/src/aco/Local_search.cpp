// ═══════════════════════════════════════════════════════════════════════════
// FILE: Local_search.cpp
//
// MỤC ĐÍCH:
//   Cải thiện nghiệm sau bước construction của ACO bằng cách di chuyển
//   các node giữa các cluster để giảm chi phí intra-cluster và đưa nghiệm
//   về trạng thái khả thi (feasible) nếu có thể.
//
// HAI LOẠI MOVE ĐƯỢC THỰC HIỆN:
//   - Relocate (Phase A): di chuyển 1 node từ cluster này sang cluster khác.
//   - Swap     (Phase B): hoán đổi cluster của 2 node thuộc 2 cluster khác nhau.
//
// CHIẾN LƯỢC CHÍNH (Phase A — Relocate):
//   Mỗi "pass" duyệt N đỉnh theo thứ tự ngẫu nhiên.
//   Với mỗi đỉnh u:
//     → Duyệt hết K cụm, tìm cụm tốt NHẤT cho u (best-of-K).
//     → Nếu score tốt nhất đó < -EPS (có cải thiện):
//         apply ngay và KẾT THÚC pass (break khỏi vòng for đỉnh).
//     → Nếu không có cải thiện: bỏ qua đỉnh này, thử đỉnh tiếp theo.
//   Nếu toàn bộ N đỉnh đều không cho cải thiện → chuyển Phase B (Swap).
//
// CHIẾN LƯỢC PHASE B (Swap):
//   Duyệt N đỉnh, mỗi đỉnh thử hoán đổi với candidate list (CL).
//   First-improvement: swap đầu tiên cải thiện → apply, kết thúc pass.
//
// ═══════════════════════════════════════════════════════════════════════════

#include "ACO.h"
#include "Local_search.h"

// Tham chiếu đến danh sách ứng viên (candidate list) toàn cục được xây trong ACO_tuned().
// globalCL[i] = danh sách GLOBAL_CL_SIZE node gần node i nhất (theo trung bình dist).
// Được dùng trong Phase B (Swap) để thu hẹp lân cận cần xét.
extern vector<vector<int>> globalCL;
extern int                 GLOBAL_CL_SIZE; // số phần tử mỗi hàng của globalCL (= 20)

// ═══════════════════════════════════════════════════════════════════════════
// HÀM CHÍNH: local_search
//
// THAM SỐ:
//   sol      : nghiệm đầu vào (được sửa trực tiếp — pass by reference)
//   rng      : bộ sinh số ngẫu nhiên Mersenne Twister 64-bit
//   maxMoves : số lượng move tối đa được phép thực hiện
//
// KẾT QUẢ:
//   sol được cập nhật thành nghiệm tốt nhất tìm được trong quá trình LS.
//   Nếu tìm được nghiệm feasible → trả nghiệm feasible tốt nhất.
//   Nếu không → trả nghiệm hiện tại với cost được tính lại.
// ═══════════════════════════════════════════════════════════════════════════
void local_search(ACOSolution &sol, mt19937_64 &rng, int maxMoves)
{
    // Nếu không được phép thực hiện move nào → thoát ngay
    if (maxMoves <= 0) return;

    // ── Tạo alias (tham chiếu) đến các thành phần của nghiệm ──
    // Mục đích: viết ngắn gọn hơn, tránh gọi sol.xxx mỗi lần.
    // Tất cả đều là reference → sửa alias = sửa trực tiếp sol.
    auto &assign    = sol.assign;                    // assign[i]    = cluster của node i (0-indexed)
    auto &members   = sol.members;                   // members[k]   = danh sách node thuộc cluster k
    auto &clusterWeight = sol.clusterWeight;         // clusterWeight[k][t] = tổng trọng số chiều t của cluster k
    auto &clusterSumDist   = sol.clusterSumDist;     // clusterSumDist[i][k]   = tổng dist(i, j) với mọi j ∈ cluster k

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 1: VIOLATION CACHE
    //
    // Thay vì tính lại vi phạm từ đầu sau mỗi move (O(N × M_weights)),
    // ta lưu vi phạm của mỗi cluster vào violCache[k] và cập nhật
    // incremental sau mỗi move → O(M_weights) mỗi lần.
    //
    // violCache[k] = tổng vi phạm ràng buộc trọng số của cluster k
    //   = Σ_t max(0, clusterWeight[k][t] - WUmat[k][t])   (vi phạm upper)
    //   + Σ_t max(0, WLmat[k][t] - clusterWeight[k][t])   (vi phạm lower)
    //
    // totalViol = Σ_k violCache[k] = tổng vi phạm toàn bộ nghiệm
    // totalViol = 0 ↔ nghiệm feasible
    // ══════════════════════════════════════════════════════════════════════

    // Hàm tính lại vi phạm của cluster k từ clusterWeight hiện tại — O(M_weights)
    // Được gọi sau mỗi move để cập nhật violCache[k] cho 2 cluster bị ảnh hưởng.
    auto recomputeViol = [&](int k) -> double {
        double v = 0.0;                        // tích lũy tổng vi phạm của cluster k
        for (int t = 0; t < M_weights; ++t) {  // duyệt mỗi chiều trọng số
            double w = clusterWeight[k][t];         // tổng trọng số chiều t của cluster k
            if (w > WUmat[k][t]) v += w - WUmat[k][t]; // vượt upper → vi phạm = lượng thừa
            if (w < WLmat[k][t]) v += WLmat[k][t] - w; // dưới lower → vi phạm = lượng thiếu
        }
        return v; // tổng vi phạm của cluster k (≥ 0)
    };

    // Khởi tạo violCache và tính totalViol ban đầu — O(K × M_weights)
    vector<double> violCache(K, 0.0); // violCache[k] = 0.0 cho tất cả cluster
    double totalViol = 0.0;           // tổng vi phạm toàn nghiệm
    for (int k = 0; k < K; ++k) {
        violCache[k] = recomputeViol(k); // tính vi phạm cluster k từ trạng thái đầu vào
        totalViol   += violCache[k];     // cộng dồn vào tổng
    }
    totalViol = max(totalViol, 0.0); // đảm bảo không âm (phòng ngừa sai số floating-point)

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 2: DUAL INCUMBENT + curDist TRACKING
    //
    // Dual incumbent (hai nghiệm lưu song song):
    //   (a) sol       : nghiệm đang làm việc (working solution)
    //                   Có thể infeasible để thuật toán thoát local optima.
    //   (b) feasBest  : snapshot nghiệm FEASIBLE tốt nhất từng tìm được
    //                   Luôn feasible (totalViol ≈ 0) và có curDist nhỏ nhất.
    //
    // curDist tracking:
    //   curDist = tổng intra-distance của sol hiện tại
    //   Thay vì tính lại O(N²) sau mỗi move, ta duy trì curDist bằng cách
    //   cộng/trừ deltaDist sau mỗi relocate/swap → O(1).
    //   curDist = Σ_i clusterSumDist[i][assign[i]]
    // ══════════════════════════════════════════════════════════════════════

    // Tính curDist một lần duy nhất từ clusterSumDist hiện tại — O(N)
    // clusterSumDist[i][assign[i]] = tổng dist(i, j) với mọi j trong cùng cluster với i
    // Σ tất cả = tổng intra-distance (mỗi cặp (i,j) được đếm 2 lần từ i và j)
    double curDist = 0.0;
    for (int i = 0; i < N; ++i)
        curDist += clusterSumDist[i][assign[i]]; // cộng dồn dist từ i đến tất cả node cùng cluster

    // feasBest: snapshot nghiệm feasible tốt nhất tìm được cho đến hiện tại
    ACOSolution feasBest;          // chứa nghiệm feasible tốt nhất (copy của sol)
    double      feasBestDist = 1e300; // intra-distance của feasBest (khởi tạo = vô cực)
    bool        feasFound    = false;  // đã tìm được ít nhất 1 nghiệm feasible chưa?

    // Nếu nghiệm đầu vào đã feasible → lưu ngay làm feasBest ban đầu
    if (totalViol < VALID_EPS) {   // VALID_EPS = epsilon nhỏ, totalViol ≈ 0 ↔ feasible
        feasFound    = true;        // đánh dấu đã có nghiệm feasible
        feasBestDist = curDist;     // lưu intra-distance
        feasBest     = sol;         // lưu snapshot toàn bộ nghiệm (deep copy)
    }

    // tryUpdateFeasBest: kiểm tra và cập nhật feasBest nếu nghiệm hiện tại tốt hơn
    // Được gọi sau mỗi move để cập nhật feasBest — O(1) nhờ curDist tracking.
    // Trả về true nếu feasBest được cập nhật (dùng để reset noFeasImprove).
    auto tryUpdateFeasBest = [&]() -> bool {
        if (totalViol >= VALID_EPS) return false; // nghiệm hiện tại infeasible → bỏ qua
        // Cập nhật nếu: chưa có feasBest, hoặc curDist nhỏ hơn feasBestDist
        if (!feasFound || curDist < feasBestDist - 1e-9) {
            feasFound    = true;       // đánh dấu đã tìm được nghiệm feasible
            feasBestDist = curDist;    // cập nhật distance tốt nhất
            feasBest     = sol;        // lưu snapshot (deep copy)
            return true;               // thông báo có cải thiện
        }
        return false; // không cải thiện feasBest
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 3: DYNAMIC PENALTY (Hệ số phạt động)
    //
    // Hàm mục tiêu trong LS: score = deltaDist + localPenalty × deltaViol
    //
    // localPenalty là hệ số phạt cho vi phạm ràng buộc — THAY ĐỔI ĐỘNG:
    //
    // Ý tưởng: khi nghiệm đang infeasible lâu → tăng penalty → ép về feasible.
    //          Khi nghiệm đã feasible và bị kẹt → giảm penalty → mở không gian
    //          tìm kiếm cho phép tạm thời vi phạm nhẹ để thoát local opt.
    //
    // Quy tắc cập nhật sau mỗi MOVE:
    //   - Feasible → penalty GIỮ NGUYÊN (đang khai thác vùng tốt)
    //   - Infeasible, mỗi gradSteps move → penalty × PEN_UP (tăng dần)

    // Quy tắc cập nhật khi KHÔNG CÓ MOVE (idle):
    //   - Feasible idle → penalty ÷ PEN_DOWN NGAY LẬP TỨC (mở rộng landscape)
    //   - Infeasible idle → penalty × PEN_UP NGAY LẬP TỨC (ép về feasible)
    //   Sau 3 pass idle liên tiếp (2 lần đổi penalty + 1 vòng kết thúc) → dừng.
    // ══════════════════════════════════════════════════════════════════════

    double localPenalty            = PENALTY_SCALE;       // bắt đầu bằng hệ số phạt toàn cục
    const double PEN_UP            = 1.5;                 // hệ số tăng penalty mỗi gradSteps move khi infeasible
    const double PEN_DOWN          = 2.0;                 // hệ số tăng penalty giảm khi kẹt tại feasible
    const double PEN_MIN           = PENALTY_SCALE * 0.2; // ngưỡng dưới: penalty không giảm dưới đây
    const double PEN_MAX           = PENALTY_SCALE * 5.0; // ngưỡng trên: penalty không tăng vượt đây
    const int    gradSteps         = 5;                   // số move giữa 2 lần tăng penalty

    int stepsInfeas  = 0; // đếm move infeasible cho lần tăng penalty tiếp theo

    // updatePenalty: gọi sau mỗi move được apply
    auto updatePenalty = [&]() {
        if (totalViol < VALID_EPS) {
            // ── Trạng thái FEASIBLE ──
            // Không thay đổi penalty: đang khai thác vùng feasible tốt
            stepsInfeas  = 0; // reset đếm steps infeasible
        } else {
            // ── Trạng thái INFEASIBLE ──
            ++stepsInfeas;  // tăng đếm move infeasible kể từ lần tăng cuối
            if (stepsInfeas >= gradSteps) {
                // Đủ gradSteps move infeasible → tăng penalty × PEN_CHANGE (tăng dần)
                localPenalty  = min(localPenalty * PEN_UP, PEN_MAX);
                stepsInfeas   = 0; // reset đếm steps
            }
        }
    };

    // updatePenaltyIdle: gọi khi toàn bộ 1 pass không tìm được move nào.
    // Thay đổi penalty NGAY LẬP TỨC (không chờ gradSteps) vì đây là tín hiệu
    // rõ ràng rằng landscape hiện tại đã cạn kiệt move.
    auto updatePenaltyIdle = [&]() {
        if (totalViol < VALID_EPS) {
            // ── Feasible + không có move ──
            // Bị kẹt tại local opt feasible → giảm penalty ngay để "mở cửa sổ",
            // cho phép tạm vi phạm nhẹ và khám phá vùng mới.
            localPenalty = max(localPenalty / PEN_DOWN, PEN_MIN);
            stepsInfeas  = 0;
        } else {
            // ── Infeasible + không có move ──
            // Vẫn infeasible nhưng bị kẹt → tăng penalty ngay để đẩy mạnh về feasible.
            localPenalty = min(localPenalty * PEN_UP, PEN_MAX);
            stepsInfeas  = 0;
        }
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 4: ĐIỀU KIỆN DỪNG (Stopping Criteria)
    //
    // Local search dừng khi thỏa MỘT TRONG CÁC điều kiện sau:
    //   [A] moveCount >= maxMoves: đã dùng hết budget move
    //   [B] noFeasImprove >= MAX_NO_FEAS_IMPROVE: đã thực hiện nhiều move
    //       mà không cải thiện được feasBest → khả năng cao đã hội tụ
    //   [C] Feasible + penalty về PEN_MIN + có ít nhất 1 pass idle:
    //       nghiệm feasible, đã giảm penalty tối đa, vẫn không có move
    //       → thuật toán hội tụ hoàn toàn
    //   [D] noMoveStreak >= PATIENCE (= 3): liên tiếp 3 pass không có move
    //       → đã thay đổi penalty 2 lần mà vẫn kẹt → dừng thuật toán
    // ══════════════════════════════════════════════════════════════════════

    // PATIENCE = 3: dừng sau 3 pass idle liên tiếp không có move.
    // Tức là: pass idle 1 → đổi penalty, pass idle 2 → đổi penalty lần 2,
    //         pass idle 3 → kết thúc (đã thay đổi penalty 2 lần mà vẫn kẹt).
    const int PATIENCE            = 3;
    const int MAX_NO_FEAS_IMPROVE = 25;            // số move tối đa không cải thiện feasBest

    int noMoveStreak  = 0; // số pass liên tiếp không có move nào được apply
    int noFeasImprove = 0; // số move liên tiếp không cải thiện feasBest

    int moveCount = 0; // tổng số move đã thực hiện (so với maxMoves)

    // Lambda shouldStop: kiểm tra tất cả điều kiện dừng — gọi trước mỗi vòng lặp
    auto shouldStop = [&]() -> bool {
        if (moveCount >= maxMoves)                         return true; // [A] hết budget
        if (noFeasImprove >= MAX_NO_FEAS_IMPROVE)          return true; // [B] không improve feasBest
        if (totalViol < VALID_EPS && localPenalty <= PEN_MIN
                                  && noMoveStreak > 0)     return true; // [C] hội tụ feasible
        if (noMoveStreak >= PATIENCE)                       return true; // [D] bị kẹt hoàn toàn
        return false; // chưa dừng
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 5: CANDIDATE LIST (Danh sách ứng viên cho Phase B — Swap)
    //
    // Trong Phase B, mỗi node u chỉ thử swap với node trong cl[u]
    // thay vì tất cả N-1 node → giảm độ phức tạp từ O(N²) xuống O(N × CL_SIZE).
    //
    // cl[u] = danh sách các node gần u nhất (theo khoảng cách trung bình).
    // Swap với node gần thường có lợi nhất về distance → CL không làm mất nhiều quality.
    //
    // Ưu tiên dùng globalCL (đã xây trong ACO_tuned, tránh tính lại).
    // Nếu globalCL không hợp lệ → xây localCL_fallback ngay trong hàm này.
    // ══════════════════════════════════════════════════════════════════════

    const vector<vector<int>> *clPtr = nullptr;  // con trỏ tới candidate list sẽ dùng
    vector<vector<int>> localCL_fallback;        // CL dự phòng (chỉ xây nếu cần)

    if (!globalCL.empty() && (int)globalCL.size() == N) {
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
    const vector<vector<int>> &cl = *clPtr; // alias tiện dùng

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 6: HÀM TIỆN ÍCH
    // ══════════════════════════════════════════════════════════════════════

    // eraseNode: xóa node khỏi danh sách members của cluster — O(|members|)
    // Dùng "swap with back + pop_back" để tránh dịch chuyển phần tử → O(1) thực tế.
    // vec: members[k], node: node cần xóa
    auto eraseNode = [](vector<int> &vec, int node) {
        auto it = find(vec.begin(), vec.end(), node); // tìm vị trí node
        if (it != vec.end()) {
            *it = vec.back(); // ghi đè bằng phần tử cuối (O(1), không giữ thứ tự)
            vec.pop_back();   // xóa phần tử cuối (đã dịch lên trên)
        }
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 7: APPLY RELOCATE — Di chuyển node u từ cluster from sang cluster to
    //
    // THAM SỐ (đã tính sẵn ở bước evaluate để tránh tính lại):
    //   u         : node cần di chuyển
    //   from      : cluster hiện tại của u
    //   to        : cluster đích
    //   deltaDist : thay đổi intra-distance (= curDist mới - curDist cũ)
    //   deltaViol : thay đổi tổng vi phạm (= totalViol mới - totalViol cũ)
    //
    // CẬP NHẬT (incremental, không tính lại từ đầu):
    //   curDist   → cộng deltaDist — O(1)
    //   assign    → gán assign[u] = to — O(1)
    //   members   → xóa u khỏi from, thêm vào to — O(|members|)
    //   clusterWeight → cộng/trừ Wmat[u][t] — O(M_weights)
    //   violCache → tính lại 2 cluster bị ảnh hưởng (from, to) — O(M_weights)
    //   totalViol → cập nhật từ violCache — O(1)
    //   clusterSumDist   → cập nhật dist từ mọi node j đến from/to — O(N)
    //
    // TỔNG ĐỘ PHỨC TẠP: O(N) (bottleneck là bước cập nhật clusterSumDist)
    // ══════════════════════════════════════════════════════════════════════

    auto applyRelocate = [&](int u, int from, int to, double deltaDist, double deltaViol) {

        // ── Cập nhật curDist — O(1) ──
        // deltaDist đã được tính sẵn trong vòng evaluate → chỉ cần cộng
        curDist += deltaDist;

        // ── Cập nhật assign và members ──
        assign[u] = to;                  // node u chuyển sang cluster to
        eraseNode(members[from], u);     // xóa u khỏi danh sách cluster from — O(|members_from|)
        members[to].push_back(u);        // thêm u vào danh sách cluster to — O(1)

        // ── Cập nhật tổng trọng số mỗi cluster — O(M_weights) ──
        // Cluster from mất đi trọng số của u; cluster to nhận thêm
        for (int t = 0; t < M_weights; ++t) {
            clusterWeight[from][t] -= Wmat[u][t]; // cluster from: bớt trọng số node u chiều t
            clusterWeight[to][t]   += Wmat[u][t]; // cluster to: thêm trọng số node u chiều t
        }

        // ── Cập nhật violCache và totalViol — O(M_weights) ──
        // Chỉ 2 cluster bị ảnh hưởng: from (mất u) và to (nhận u)
        double oldVFrom = violCache[from]; // lưu vi phạm cũ của cluster from
        double oldVTo   = violCache[to];   // lưu vi phạm cũ của cluster to
        violCache[from] = recomputeViol(from);   // tính lại vi phạm mới của cluster from
        violCache[to]   = recomputeViol(to);     // tính lại vi phạm mới của cluster to
        // Cập nhật totalViol bằng cách trừ giá trị cũ, cộng giá trị mới
        totalViol += (violCache[from] - oldVFrom) + (violCache[to] - oldVTo);
        totalViol  = max(totalViol, 0.0); // đảm bảo không âm (sai số floating-point)

        // ── Cập nhật clusterSumDist cho mọi node v ≠ u — O(N) ──
        // Khi u rời cluster from và vào cluster to:
        //   clusterSumDist[v][from] giảm đi dist(v, u) (u không còn trong from)
        //   clusterSumDist[v][to]   tăng lên dist(v, u) (u mới gia nhập to)
        for (int v = 0; v < N; ++v) {
            if (v == u) continue;              // bỏ qua node u chính nó
            double d = distmat[v][u];    // khoảng cách từ v đến u
            clusterSumDist[v][from] -= d;             // v mất 1 neighbor (u) trong cluster from
            clusterSumDist[v][to]   += d;             // v có thêm 1 neighbor (u) trong cluster to
        }
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 8: APPLY SWAP — Hoán đổi cluster của 2 node u và v
    //
    // Điều kiện: u ∈ cluster cu, v ∈ cluster cv, cu ≠ cv
    // Sau swap: u ∈ cv, v ∈ cu
    //
    // THAM SỐ (đã tính sẵn ở bước evaluate):
    //   u, v      : 2 node cần hoán đổi
    //   deltaDist : thay đổi intra-distance
    //   deltaViol : thay đổi tổng vi phạm
    //
    // CẬP NHẬT tương tự applyRelocate nhưng cho 2 node cùng lúc:
    //   curDist, assign, members, clusterWeight, violCache, totalViol, clusterSumDist
    //
    // TỔNG ĐỘ PHỨC TẠP: O(N) (bottleneck vẫn là clusterSumDist)
    // ══════════════════════════════════════════════════════════════════════

    auto applySwap = [&](int u, int v, double deltaDist, double deltaViol) {
        int cu = assign[u]; // cluster hiện tại của u
        int cv = assign[v]; // cluster hiện tại của v

        // ── Cập nhật curDist — O(1) ──
        curDist += deltaDist; // deltaDist đã tính sẵn

        // ── Cập nhật assign và members ──
        assign[u] = cv;                          // u chuyển sang cluster cv
        assign[v] = cu;                          // v chuyển sang cluster cu
        eraseNode(members[cu], u);               // xóa u khỏi cu
        eraseNode(members[cv], v);               // xóa v khỏi cv
        members[cv].push_back(u);                // thêm u vào cv
        members[cu].push_back(v);                // thêm v vào cu

        // ── Cập nhật tổng trọng số mỗi cluster — O(M_weights) ──
        // Cluster cu: mất u nhưng nhận v → net = Wmat[v][t] - Wmat[u][t]
        // Cluster cv: mất v nhưng nhận u → net = Wmat[u][t] - Wmat[v][t]
        for (int t = 0; t < M_weights; ++t) {
            clusterWeight[cu][t] += Wmat[v][t] - Wmat[u][t]; // cu: đổi u lấy v
            clusterWeight[cv][t] += Wmat[u][t] - Wmat[v][t]; // cv: đổi v lấy u
        }

        // ── Cập nhật violCache và totalViol — O(M_weights) ──
        double oldVCu = violCache[cu]; // vi phạm cũ cluster cu
        double oldVCv = violCache[cv]; // vi phạm cũ cluster cv
        violCache[cu] = recomputeViol(cu);   // tính lại sau khi hoán đổi
        violCache[cv] = recomputeViol(cv);   // tính lại sau khi hoán đổi
        totalViol += (violCache[cu] - oldVCu) + (violCache[cv] - oldVCv);
        totalViol  = max(totalViol, 0.0); // đảm bảo không âm

        // ── Cập nhật clusterSumDist cho mọi node w ≠ u, v — O(N) ──
        // Từ góc nhìn của node a bất kỳ:
        //   clusterSumDist[a][cu]: mất u (rời đi) nhưng nhận v (mới đến) → thay đổi = dv - du
        //   clusterSumDist[a][cv]: mất v (rời đi) nhưng nhận u (mới đến) → thay đổi = du - dv
        for (int a = 0; a < N; ++a) {
            if (a == u || a == v) continue;    // bỏ qua u và v (tính riêng bên dưới)
            double dau = distmat[a][u];    // dist từ a đến u
            double dav = distmat[a][v];    // dist từ a đến v
            clusterSumDist[a][cu] += dav - dau;          // cu: mất u, nhận v → +dv -du
            clusterSumDist[a][cv] += dau - dav;          // cv: mất v, nhận u → +du -dv
        }

        // ── Cập nhật clusterSumDist của u và v với cả 2 cluster — O(N/K) ──
        // Sau hoán đổi, u và v nằm ở cluster khác → phải tính lại clusterSumDist với cu và cv
        // cu mất u nhận v → clusterSumDist[u][cu] += dist(u,v)
        // cv mất v nhận u → clusterSumDist[u][cv] -= dist(u,v)
        const double duv = distmat[u][v];
        const double dvu = distmat[v][u];
        clusterSumDist[u][cu] += duv;
        clusterSumDist[u][cv] -= duv;
        clusterSumDist[v][cv] += dvu;
        clusterSumDist[v][cu] -= dvu;
    };

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 9: VÒNG LẶP CHÍNH
    //
    // Cấu trúc: nhiều "pass", mỗi pass gồm 2 phase:
    //
    // PHASE A — Relocate (best-of-K per node, first-improve per pass):
    //   Shuffle thứ tự N đỉnh.
    //   Với từng đỉnh u theo thứ tự:
    //     Duyệt hết K cụm → tìm cụm tốt NHẤT (score = deltaDist + pen × deltaViol).
    //     Nếu score tốt nhất < -EPS → apply move đó + kết thúc pass ngay (first-improve).
    //     Nếu không cải thiện → bỏ qua u, thử đỉnh tiếp theo.
    //   Nếu hết N đỉnh mà không có cải thiện → sang Phase B.
    //
    // PHASE B — Swap (first-improvement với Candidate List):
    //   Với từng đỉnh u (cùng nodeOrder từ Phase A):
    //     Thử swap u với mỗi v trong cl[u] (candidate list — node gần nhất).
    //     Swap đầu tiên cải thiện → apply, kết thúc phase B.
    //     Fallback: thử EXTRA_RANDOM cặp ngẫu nhiên nếu CL không cho move.
    //   Nếu không có swap → tăng noMoveStreak, gọi updatePenaltyIdle.
    // ══════════════════════════════════════════════════════════════════════

    const double SCORE_EPS = 1e-9;          // ngưỡng cải thiện: score < -SCORE_EPS → chấp nhận
    const int EXTRA_RANDOM = 10;    // số cặp ngẫu nhiên thử thêm trong fallback của Phase B

    // Thứ tự duyệt đỉnh (xáo trộn mỗi pass để tránh bias)
    vector<int> nodeOrder(N);
    iota(nodeOrder.begin(), nodeOrder.end(), 0); // khởi tạo [0, 1, 2, ..., N-1]

    // Phân phối ngẫu nhiên chọn node cho fallback swap
    uniform_int_distribution<int> randNode(0, N - 1);

    // ── Vòng lặp ngoài: mỗi vòng = 1 pass (Phase A + Phase B nếu cần) ──
    while (!shouldStop()) // lặp cho đến khi thỏa điều kiện dừng
    {
        // Xáo trộn thứ tự duyệt đỉnh mỗi pass → tránh bias (luôn bắt đầu từ cùng 1 đỉnh)
        shuffle(nodeOrder.begin(), nodeOrder.end(), rng);

        bool improved = false; // đánh dấu: pass này đã apply được move nào chưa?

        // ══════════════════════════════════════════════════════════════════
        // PHASE A: RELOCATE — Tìm best-of-K, first-improve per pass
        //
        // Với mỗi đỉnh u (theo nodeOrder):
        //   1. Tính score cho mỗi cluster to ≠ from.
        //      score(to) = deltaDist(u, from→to) + localPenalty × deltaViol(u, from→to)
        //   2. Lưu cluster có score nhỏ nhất (bestTo, bestScore).
        //   3. Nếu bestScore < -SCORE_EPS → apply ngay, kết thúc pass.
        //   4. Nếu không → thử đỉnh tiếp theo.
        //
        // Mục tiêu: giảm score (= giảm chi phí và/hoặc vi phạm) → chọn cluster tốt nhất
        // cho u thay vì chọn cluster đầu tiên cho cải thiện (best-of-K per node).
        // ══════════════════════════════════════════════════════════════════

        // Duyệt N đỉnh, dừng sớm khi tìm được move đầu tiên
        for (int ii = 0; ii < N && !shouldStop() && !improved; ++ii)
        {
            const int u    = nodeOrder[ii]; // đỉnh đang xét (theo thứ tự đã shuffle)
            const int from = assign[u];     // cluster hiện tại của u

            // Không relocate nếu cluster chỉ có 1 node → sẽ tạo cluster rỗng (không hợp lệ)
            if ((int)members[from].size() <= 1) continue;

            // Khởi tạo ứng viên tốt nhất cho u
            int    bestTo    = -1;         // cluster đích tốt nhất
            double bestScore = -SCORE_EPS; // score âm tốt nhất (giảm hàm chi phí tổng)
            double bestDelta = 0.0;        // deltaDist của bestTo (dùng khi apply)
            double bestDViol = 0.0;        // deltaViol của bestTo (dùng khi apply)

            // Duyệt hết K cluster để tìm cluster đích tốt nhất cho u
            for (int to = 0; to < K; ++to)
            {
                if (to == from) continue; // không thể chuyển sang cluster đang ở

                // ── Tính deltaDist(u, from→to) ──
                // Sau khi chuyển u từ from sang to:
                //   intra-dist từ u với cluster from = clusterSumDist[u][from] (u đang ở from)
                //   intra-dist từ u với cluster to   = clusterSumDist[u][to] (u gia nhập to)
                // Chênh lệch = clusterSumDist[u][to] - clusterSumDist[u][from]
                const double deltaDist = clusterSumDist[u][to] - clusterSumDist[u][from];

                // ── Tính deltaViol(u, from→to) ──
                // Tính tổng vi phạm MỚI của 2 cluster bị ảnh hưởng (from và to)
                // sau khi giả sử u được chuyển
                double vfAfter = 0.0; // vi phạm mới của cluster from (sau khi mất u)
                double vtAfter = 0.0; // vi phạm mới của cluster to   (sau khi nhận u)
                for (int t = 0; t < M_weights; ++t) {
                    const double sf = clusterWeight[from][t] - Wmat[u][t]; // tổng W của from sau khi mất u
                    const double st = clusterWeight[to][t]   + Wmat[u][t]; // tổng W của to sau khi nhận u
                    // Vi phạm lower bound của cluster from
                    if      (sf < WLmat[from][t]) vfAfter += WLmat[from][t] - sf;
                    // Vi phạm upper bound của cluster from
                    else if (sf > WUmat[from][t]) vfAfter += sf - WUmat[from][t];
                    // Vi phạm lower bound của cluster to
                    if      (st < WLmat[to][t])   vtAfter += WLmat[to][t]   - st;
                    // Vi phạm upper bound của cluster to
                    else if (st > WUmat[to][t])   vtAfter += st - WUmat[to][t];
                }
                // deltaViol = (vi phạm mới của from + to) - (vi phạm cũ của from + to)
                const double deltaViol = (vfAfter + vtAfter)
                                       - (violCache[from] + violCache[to]);

                // ── Tính score tổng hợp ──
                // score < 0 → move cải thiện hàm mục tiêu (tốt hơn)
                // score gộp cả distance và penalty vi phạm với hệ số localPenalty
                const double score = deltaDist + localPenalty * deltaViol;

                // Cập nhật best nếu cluster to tốt hơn bestTo hiện tại
                if (score < bestScore) { // score nhỏ hơn → tốt hơn
                    bestScore = score;   // cập nhật score tốt nhất
                    bestTo    = to;      // lưu cluster đích
                    bestDelta = deltaDist; // lưu deltaDist để dùng khi apply
                    bestDViol = deltaViol; // lưu deltaViol để dùng khi apply
                }
            }
            // Kết thúc duyệt K cluster cho node u

            // Nếu tìm được cluster cải thiện (bestTo >= 0) → apply move và kết thúc pass
            if (bestTo >= 0) {
                applyRelocate(u, from, bestTo, bestDelta, bestDViol);
                ++moveCount;     // tăng đếm move
                improved = true; // đánh dấu pass này có move → sẽ bắt đầu pass mới

                updatePenalty(); // cập nhật penalty dựa trên trạng thái hiện tại

                // Kiểm tra và cập nhật feasBest
                if (tryUpdateFeasBest()) noFeasImprove = 0;  // cải thiện feasBest → reset đếm
                else if (feasFound)      ++noFeasImprove;    // không cải thiện → tăng đếm
                // improved = true → vòng for dừng sớm nhờ điều kiện !improved
            }
        }
        // Kết thúc Phase A

        // Nếu Phase A tìm được move → reset noMoveStreak và bắt đầu pass mới
        if (improved) {
            noMoveStreak = 0; // reset đếm pass idle
            continue;         // bắt đầu pass mới (shuffle lại nodeOrder)
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE B: SWAP — First-improvement với Candidate List
        //
        // Chỉ chạy khi Phase A không tìm được move nào.
        //
        // Swap(u, v): hoán đổi cluster của u và v (assign[u] ↔ assign[v])
        //   Điều kiện: assign[u] ≠ assign[v] (2 cluster khác nhau)
        //   score = deltaDist + localPenalty × deltaViol
        //   Chỉ chấp nhận nếu score < -SCORE_EPS (cải thiện)
        //
        // Thứ tự thử swap:
        //   1. Với mỗi u theo nodeOrder → thử swap với v ∈ cl[u] (candidate list)
        //      → node gần nhau về khoảng cách → swap thường có lợi nhất về distance
        //   2. Fallback: thử EXTRA_RANDOM cặp (u, v) ngẫu nhiên
        //      → mở rộng lân cận sang node xa (mà CL bỏ qua)
        //      → phòng trường hợp CL không đủ để thoát local opt
        //
        // First-improvement: swap đầu tiên cải thiện → apply ngay, kết thúc phase.
        // ══════════════════════════════════════════════════════════════════

        // Duyệt N đỉnh theo cùng nodeOrder với Phase A
        for (int ii = 0; ii < N && !shouldStop() && !improved; ++ii)
        {
            const int u  = nodeOrder[ii]; // đỉnh đang xét
            const int cu = assign[u];     // cluster hiện tại của u

            // ── Lambda nội bộ: thử swap u ↔ v ──
            // Tính score, kiểm tra cải thiện, apply nếu đạt ngưỡng.
            // Trả về true nếu swap được apply (dừng tìm kiếm thêm).
            auto trySwap = [&](int v) -> bool {
                if (v == u) return false;   // không swap node với chính nó
                const int cv = assign[v];   // cluster của v
                if (cu == cv) return false; // không swap 2 node cùng cluster

                // ── Tính deltaDist(swap u↔v) ──
                // Sau khi u→cv và v→cu:
                //   u mất dist với cu-members, nhận dist với cv-members
                //   v mất dist với cv-members, nhận dist với cu-members
                //   Tuy nhiên phải trừ đi dist(u,v) đã tính 2 lần (từ clusterSumDist[u] và clusterSumDist[v])
                //   và cộng lại đúng 1 lần (dist(u,v) trong intra-cluster mới)
                //   Công thức đầy đủ:
                //     deltaDist = (clusterSumDist[u][cv] - clusterSumDist[u][cu])   (u chuyển từ cu→cv)
                //               + (clusterSumDist[v][cu] - clusterSumDist[v][cv])   (v chuyển từ cv→cu)
                //               - distmat[u][v] - distmat[v][u]       (loại bỏ đếm đôi)
                const double deltaDist =
                    (clusterSumDist[u][cv] - clusterSumDist[u][cu])
                  + (clusterSumDist[v][cu] - clusterSumDist[v][cv])
                  - distmat[u][v] - distmat[v][u];

                // ── Tính deltaViol(swap u↔v) ──
                // Tính vi phạm MỚI của cluster cu (mất u, nhận v) và cv (mất v, nhận u)
                double vcuAfter = 0.0; // vi phạm mới cluster cu
                double vcvAfter = 0.0; // vi phạm mới cluster cv
                for (int t = 0; t < M_weights; ++t) {
                    const double scu = clusterWeight[cu][t] - Wmat[u][t] + Wmat[v][t]; // W mới của cu
                    const double scv = clusterWeight[cv][t] - Wmat[v][t] + Wmat[u][t]; // W mới của cv
                    if      (scu < WLmat[cu][t]) vcuAfter += WLmat[cu][t] - scu; // vi phạm lower cu
                    else if (scu > WUmat[cu][t]) vcuAfter += scu - WUmat[cu][t]; // vi phạm upper cu
                    if      (scv < WLmat[cv][t]) vcvAfter += WLmat[cv][t] - scv; // vi phạm lower cv
                    else if (scv > WUmat[cv][t]) vcvAfter += scv - WUmat[cv][t]; // vi phạm upper cv
                }
                // deltaViol = (vi phạm mới của cu + cv) - (vi phạm cũ của cu + cv)
                const double deltaViol = (vcuAfter + vcvAfter)
                                       - (violCache[cu] + violCache[cv]);
                // Tính score
                const double score     = deltaDist + localPenalty * deltaViol;

                if (score >= -SCORE_EPS) return false; // không cải thiện → bỏ qua

                // Có cải thiện → apply swap
                applySwap(u, v, deltaDist, deltaViol);
                ++moveCount;     // tăng đếm move
                improved = true; // đánh dấu phase B tìm được move

                updatePenalty(); // cập nhật penalty
                // Cập nhật feasBest nếu nghiệm hiện tại tốt hơn
                if (tryUpdateFeasBest()) noFeasImprove = 0;
                else if (feasFound)      ++noFeasImprove;

                return true; // thông báo đã apply swap
            };

            // ── Thử swap u với mỗi v trong cl[u] (candidate list) ──
            // cl[u] chứa GLOBAL_CL_SIZE node gần u nhất → ưu tiên vì swap thường có lợi nhất
            bool foundSwap = false;
            for (int v : cl[u]) {
                if (shouldStop()) break;              // kiểm tra điều kiện dừng
                if (trySwap(v)) { foundSwap = true; break; } // first-improvement: dừng ngay
            }

            // ── Fallback: thử EXTRA_RANDOM cặp ngẫu nhiên ──
            // Chỉ chạy nếu CL không tìm được swap.
            // Mở rộng lân cận sang node xa (ngoài CL) → thoát local opt mà CL bỏ lỡ.
            if (!foundSwap && !shouldStop()) {
                for (int r = 0; r < EXTRA_RANDOM; ++r) {
                    if (shouldStop()) break;
                    if (trySwap(randNode(rng))) break; // first-improvement: dừng ngay khi tìm được
                }
            }
        }
        // Kết thúc Phase B

        // Cập nhật trạng thái sau 1 pass đầy đủ (Phase A + Phase B)
        if (improved) {
            noMoveStreak = 0; // có move → reset đếm pass idle
        } else {
            // Không có move nào trong cả Phase A lẫn Phase B
            ++noMoveStreak;       // tăng đếm pass idle (dùng cho điều kiện dừng PATIENCE)
            updatePenaltyIdle(); // cập nhật penalty theo trạng thái idle
        }
    }
    // ── Kết thúc vòng lặp chính ──

    // ══════════════════════════════════════════════════════════════════════
    // BƯỚC 10: TRẢ VỀ KẾT QUẢ
    //
    // Chiến lược:
    //   Ưu tiên LUÔN trả nghiệm feasible tốt nhất từng tìm được (feasBest),
    //   ngay cả khi working solution (sol) hiện tại có intra-distance nhỏ hơn
    //   nhưng infeasible — vì nghiệm infeasible không được chấp nhận trong ACO.
    //
    //   Nếu không tìm được nghiệm feasible nào → trả nghiệm hiện tại (sol)
    //   với cost được tính lại bằng compute_cost_fast để đồng bộ với phần
    //   còn lại của ACO (bao gồm cả penalty vi phạm).
    // ══════════════════════════════════════════════════════════════════════

    if (feasFound) {
        // Trả nghiệm feasible tốt nhất tìm được
        sol          = feasBest;      // phục hồi snapshot feasBest (deep copy ngược lại)
        sol.feasible = true;          // đánh dấu feasible
        sol.cost     = feasBestDist;  // cost = intra-distance thuần (không có penalty)
    } else {
        // Không tìm được nghiệm feasible → trả nghiệm working hiện tại
        sol.feasible = false;                     // đánh dấu infeasible
        sol.cost     = compute_cost_fast(sol);    // tính lại cost (intra + penalty) để đồng bộ ACO
    }
}
// ─── KẾT THÚC FILE Local_search.cpp ───
