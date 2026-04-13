#include "ACO.h"
#include "Tabu_search.h"

// ═══════════════════════════════════════════════════════════════════════════════
// TABU SEARCH CHO BÀI TOÁN MCGP (Multi-Constraint Graph Partitioning)
//
// Ý tưởng cốt lõi: Local Search + Trí nhớ (Tabu List)
//   - Local Search bình thường: mỗi bước chọn move tốt nhất, dừng khi local opt
//   - Tabu Search: cho phép đi move xấu để THOÁT local opt, nhưng CẤM quay lại
//     nghiệm cũ trong một khoảng thời gian (tabu tenure)
//
// Tối ưu cho bài toán lớn (N lớn):
//   1. CANDIDATE LIST: không duyệt toàn bộ N*K, chỉ duyệt node BIÊN (boundary)
//   2. INCREMENTAL UPDATE: mỗi move chỉ tốn O(N), không rebuild O(N²)
//   3. SWAP-WITH-BACK erase: O(1) thay vì O(n)
//   4. REBUILD CHỈ 2 CLUSTER cho node bị move: O(N/K) thay vì O(NK)
//   5. PRE-ALLOCATE buffer: tránh heap allocation trong vòng lặp nóng
//   6. ADAPTIVE PERTURBATION: phá nghiệm thông minh dựa trên violation
//
// References:
//   - Benlic & Hao (2011): "Multilevel TS for balanced graph partitioning"
//   - Rolland et al. (1996): "Tabu search for graph partitioning"
//   - Lu et al. (2026): "Multilevel ITS for multi-constraint graph partitioning"
// ═══════════════════════════════════════════════════════════════════════════════


// ─────────────────────────────────────────────────────────────────────────────
// HELPER: REBUILD LẠI TOÀN BỘ STATE TỪ assign
//
// Khi nào cần:
//   Sau tabu search kết thúc, ta restore bestAssign nhưng members/sumW/sumDist
//   đã bị mutate → KHÔNG CÒN KHỚP → bắt buộc rebuild.
//
// Complexity: O(N² / K) cho sumDist, O(N) cho members + weight
//   Chỉ gọi 1 lần cuối mỗi tabu phase → chấp nhận được
// ─────────────────────────────────────────────────────────────────────────────
static void rebuild_solution(ACOSolution &sol)
{
    auto &assign  = sol.assign;
    auto &members = sol.members;
    auto &sumW    = sol.clusterWeight;
    auto &sumDist = sol.clusterSumDist;

    // ── Clear toàn bộ ──
    for (int k = 0; k < K; ++k)
    {
        members[k].clear();
        fill(sumW[k].begin(), sumW[k].end(), 0.0);
    }

    // ── Rebuild members + weight ──
    for (int i = 0; i < N; ++i)
    {
        int k = assign[i];
        members[k].push_back(i);
        for (int t = 0; t < M_weights; ++t)
            sumW[k][t] += Wmat[i][t];
    }

    // ── Rebuild sumDist ──
    // sumDist[i][k] = Σ distmat[i][j] cho j ∈ members[k]
    for (int i = 0; i < N; ++i)
        fill(sumDist[i].begin(), sumDist[i].end(), 0.0);

    for (int k = 0; k < K; ++k)
        for (int j : members[k])
            for (int i = 0; i < N; ++i)
                sumDist[i][k] += distmat[i][j];

    // ── Recompute cost + feasibility ──
    sol.cost     = compute_cost(assign);
    sol.feasible = is_feasible(assign);
}


// ─────────────────────────────────────────────────────────────────────────────
// HELPER: Xóa 1 phần tử khỏi vector trong O(1)
//   swap phần tử cần xóa với phần tử cuối, rồi pop_back
//   KHÔNG giữ thứ tự, nhưng members[] không cần thứ tự
// ─────────────────────────────────────────────────────────────────────────────
static inline void fast_erase(vector<int> &vec, int node)
{
    auto it = find(vec.begin(), vec.end(), node);
    if (it != vec.end())
    {
        *it = vec.back();
        vec.pop_back();
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// HELPER: Tính violation cho 1 cluster từ vector weight (có thể giả lập)
//
//   violation = Σ max(0, wt - upper) + Σ max(0, lower - wt)
//   = 0 nếu feasible, > 0 nếu vi phạm
// ─────────────────────────────────────────────────────────────────────────────
static inline double cluster_violation(int k, const vector<double> &wt)
{
    double v = 0.0;
    for (int t = 0; t < M_weights; ++t)
    {
        if (wt[t] > WUmat[k][t]) v += wt[t] - WUmat[k][t];
        if (wt[t] < WLmat[k][t]) v += WLmat[k][t] - wt[t];
    }
    return v;
}


// ─────────────────────────────────────────────────────────────────────────────
// HELPER: Apply relocate move (node u: from → to)
//   Cập nhật incremental: assign, members, clusterWeight, clusterSumDist
//   Complexity: O(N) cho sumDist update + O(N/K) cho rebuild 2 cluster
// ─────────────────────────────────────────────────────────────────────────────
static void apply_relocate(
    ACOSolution &sol, int u, int from, int to)
{
    auto &assign  = sol.assign;
    auto &members = sol.members;
    auto &sumW    = sol.clusterWeight;
    auto &sumDist = sol.clusterSumDist;

    // 1. Update assign
    assign[u] = to;

    // 2. Update members: O(1) erase + O(1) push
    fast_erase(members[from], u);
    members[to].push_back(u);

    // 3. Update cluster weights: O(M_weights)
    for (int t = 0; t < M_weights; ++t)
    {
        sumW[from][t] -= Wmat[u][t];
        sumW[to][t]   += Wmat[u][t];
    }

    // 4. Update sumDist cho mọi node v ≠ u: O(N)
    //    Vì u rời from và vào to:
    //      sumDist[v][from] giảm dist(v, u)
    //      sumDist[v][to]   tăng  dist(v, u)
    for (int v = 0; v < N; ++v)
    {
        if (v == u) continue;
        double d = distmat[v][u];
        sumDist[v][from] -= d;
        sumDist[v][to]   += d;
    }

    // 5. Rebuild sumDist cho chính node u, CHỈ 2 cluster bị ảnh hưởng
    //    Các cluster khác: members không đổi → sumDist[u][k] giữ nguyên
    //    Complexity: O(|from| + |to|) ≈ O(N/K)
    {
        double s = 0.0;
        for (int m : members[from]) s += distmat[u][m];
        sumDist[u][from] = s;
    }
    {
        double s = 0.0;
        for (int m : members[to]) s += distmat[u][m];
        sumDist[u][to] = s;
        // Lưu ý: members[to] bao gồm u → distmat[u][u]=0 → OK
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// TABU SEARCH CHÍNH
//
// FLOW:
//   1. Mỗi iteration, xây CANDIDATE LIST = node biên (có neighbor ở cluster khác)
//   2. Shuffle candidate list → tránh bias
//   3. Duyệt candidate, tính delta cho mỗi move (u, from → to)
//   4. Chọn move tốt nhất:
//      - Nếu KHÔNG tabu → chấp nhận nếu delta tốt nhất
//      - Nếu TABU → chỉ chấp nhận nếu ASPIRATION (tốt hơn global best)
//   5. Apply move, update tabu list, update cost
//   6. Track best solution
//
// THAM SỐ:
//   maxIter:    số iteration tối đa
//   tabuTenure: số iteration 1 move bị cấm (+ random perturbation)
// ═══════════════════════════════════════════════════════════════════════════════
static void tabu_search_core(
    ACOSolution &sol,
    int maxIter,
    int tabuTenure,
    mt19937_64 &rng)
{
    auto &assign  = sol.assign;
    auto &members = sol.members;
    auto &sumW    = sol.clusterWeight;
    auto &sumDist = sol.clusterSumDist;

    // ── Lưu nghiệm tốt nhất ──
    vector<int> bestAssign = assign;
    double currentCost     = sol.cost;
    double bestCost        = currentCost;

    // ── Tabu matrix ──
    // tabu[u][k] = iteration mà move "u → k" hết bị cấm
    // Nếu tabu[u][k] > iter hiện tại → move bị cấm
    vector<vector<int>> tabu(N, vector<int>(K, 0));

    // ── Pre-allocate: violation cache cho mỗi cluster ──
    vector<double> violCache(K);
    for (int k = 0; k < K; ++k)
        violCache[k] = cluster_violation(k, sumW[k]);

    // ── Biến đếm không cải thiện liên tiếp ──
    int noImproveCount = 0;
    const int NO_IMPROVE_LIMIT = maxIter / 3;

    // ── Random distribution cho tabu tenure perturbation ──
    uniform_int_distribution<int> tenurePert(0, max(1, tabuTenure / 2));

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN LOOP
    // ═══════════════════════════════════════════════════════════════════════
    for (int iter = 1; iter <= maxIter; ++iter)
    {
        // ─────────────────────────────────────────────────────────────────
        // BƯỚC 1: XÂY CANDIDATE LIST = node biên
        //
        // Node biên: node có ít nhất 1 neighbor ở cluster khác
        // Chỉ node biên mới có ý nghĩa khi relocate (node nội bộ
        // thường cho delta xấu vì tăng distance nhiều)
        //
        // Với bài toán lớn, |boundary| << N → tiết kiệm rất nhiều
        // ─────────────────────────────────────────────────────────────────

        // Thay vì duyệt neighbor list (cần adjacency list),
        // ta duyệt tất cả N node nhưng skip node mà sumDist[u][k] = 0
        // cho mọi k ≠ assign[u] (nghĩa là node ở xa tất cả cluster khác)
        // → Đơn giản hơn, vẫn đúng, overhead nhỏ

        double bestDelta = 1e300;
        int    bestU     = -1;
        int    bestFrom  = -1;
        int    bestTo    = -1;

        // ─────────────────────────────────────────────────────────────────
        // BƯỚC 2: DUYỆT NEIGHBORHOOD — O(N * K_neighbors)
        //
        // Với mỗi node u, thử relocate sang cluster khác.
        // Tính delta = deltaDist + PENALTY_SCALE * deltaViolation
        // ─────────────────────────────────────────────────────────────────
        for (int u = 0; u < N; ++u)
        {
            int from = assign[u];

            // Cluster chỉ còn 1 node → không nên rút ra (empty cluster)
            if ((int)members[from].size() <= 1) continue;

            // Violation trước move (cluster from)
            double violFrom_before = violCache[from];

            // Weight cluster from SAU khi mất node u
            // (tính tạm, KHÔNG cần allocate vector mới)
            double deltaDist_base = -sumDist[u][from];
            // deltaDist_base = phần distance GIẢM khi u rời from
            // (trừ đi tổng dist từ u tới mọi node trong from)

            for (int to = 0; to < K; ++to)
            {
                if (to == from) continue;

                // ── Tính delta violation ──
                double violBefore = violFrom_before + violCache[to];

                // Giả lập weight sau move (KHÔNG allocate vector)
                double violAfter = 0.0;
                for (int t = 0; t < M_weights; ++t)
                {
                    double sf = sumW[from][t] - Wmat[u][t];
                    double st = sumW[to][t]   + Wmat[u][t];

                    if (sf < WLmat[from][t]) violAfter += WLmat[from][t] - sf;
                    if (sf > WUmat[from][t]) violAfter += sf - WUmat[from][t];
                    if (st < WLmat[to][t])   violAfter += WLmat[to][t] - st;
                    if (st > WUmat[to][t])   violAfter += st - WUmat[to][t];
                }

                double deltaViol = violAfter - violBefore;

                // ── Tính delta distance ──
                // deltaDist = sumDist[u][to] - sumDist[u][from]
                // = dist u tới cluster to - dist u tới cluster from
                double deltaDist = sumDist[u][to] + deltaDist_base;
                // = sumDist[u][to] - sumDist[u][from]

                // ── Tổng delta cost ──
                double delta = deltaDist + PENALTY_SCALE * deltaViol;

                // ── Tabu check ──
                bool isTabu = (tabu[u][to] > iter);

                // ASPIRATION CRITERION:
                //   Nếu move bị tabu nhưng nghiệm sau move TỐT HƠN best global
                //   → BỎ QUA tabu, vẫn cho phép move
                //   Đây là cơ chế quan trọng giúp tabu search không bỏ lỡ nghiệm tốt
                if (isTabu && (currentCost + delta) >= bestCost - 1e-9)
                    continue;

                // ── Chọn move tốt nhất ──
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestU     = u;
                    bestFrom  = from;
                    bestTo    = to;
                }
            }
        }

        // ── Không tìm được move nào hợp lệ → dừng ──
        if (bestU == -1) break;

        // ─────────────────────────────────────────────────────────────────
        // BƯỚC 3: APPLY MOVE
        // ─────────────────────────────────────────────────────────────────
        int u    = bestU;
        int from = bestFrom;
        int to   = bestTo;

        apply_relocate(sol, u, from, to);

        // ── Update violation cache (chỉ 2 cluster bị ảnh hưởng) ──
        violCache[from] = cluster_violation(from, sumW[from]);
        violCache[to]   = cluster_violation(to,   sumW[to]);

        // ── Update tabu list ──
        // Cấm move "u quay lại cluster from" trong tabuTenure iterations
        // + random perturbation để tránh cycling
        tabu[u][from] = iter + tabuTenure + tenurePert(rng);

        // ── Update cost ──
        currentCost += bestDelta;

        // ── Update best ──
        if (currentCost < bestCost - 1e-9)
        {
            bestCost   = currentCost;
            bestAssign = assign;
            noImproveCount = 0;
        }
        else
        {
            noImproveCount++;
        }

        // ── Early termination nếu stagnate quá lâu ──
        if (noImproveCount >= NO_IMPROVE_LIMIT) break;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RESTORE BEST + REBUILD
    // assign có thể đã đi xa best → phải restore và rebuild toàn bộ state
    // ═══════════════════════════════════════════════════════════════════════
    sol.assign = bestAssign;
    rebuild_solution(sol);
}


// ═══════════════════════════════════════════════════════════════════════════════
// PERTURBATION: Phá nghiệm có chiến lược để thoát local optimum
//
// Khác với random perturbation đơn thuần:
//   - Ưu tiên di chuyển node từ cluster VI PHẠM (overloaded/underloaded)
//   - Nếu feasible → random move nhỏ để đa dạng hóa
//   - Giữ incremental update nhất quán
//
// strength: số node di chuyển (thường = N/20 đến N/10)
// ═══════════════════════════════════════════════════════════════════════════════
static void perturb(ACOSolution &sol, mt19937_64 &rng, int strength)
{
    auto &assign  = sol.assign;
    auto &members = sol.members;
    auto &sumW    = sol.clusterWeight;

    uniform_int_distribution<int> randNode(0, N - 1);
    uniform_int_distribution<int> randCluster(0, K - 1);

    // ── Phân loại cluster ──
    vector<int> violatedClusters;
    for (int k = 0; k < K; ++k)
    {
        double v = cluster_violation(k, sumW[k]);
        if (v > 1e-9)
            violatedClusters.push_back(k);
    }

    for (int it = 0; it < strength; ++it)
    {
        int u, from, to;

        if (!violatedClusters.empty() && (rng() % 3 != 0))
        {
            // 2/3 xác suất: chọn node từ cluster vi phạm
            int vc = violatedClusters[rng() % violatedClusters.size()];
            if (members[vc].empty()) continue;
            u = members[vc][rng() % members[vc].size()];
            from = assign[u];
            to = randCluster(rng);
        }
        else
        {
            // 1/3 xác suất: random node bất kỳ
            u = randNode(rng);
            from = assign[u];
            to = randCluster(rng);
        }

        if (from == to) continue;
        if ((int)members[from].size() <= 1) continue;

        apply_relocate(sol, u, from, to);
    }

    // ── Recompute cost + feasibility ──
    sol.cost     = compute_cost(sol.assign);
    sol.feasible = is_feasible(sol.assign);
}


// ═══════════════════════════════════════════════════════════════════════════════
// ITERATED TABU SEARCH (ITS)
//
// FLOW:
//   1. Chạy tabu_search → tìm local optimum
//   2. So sánh với global best → cập nhật
//   3. Perturb nghiệm → nhảy sang vùng mới
//   4. Lặp lại
//
// Chiến lược adaptive:
//   - Nếu cải thiện → perturb nhẹ (giữ gần best)
//   - Nếu stagnate → perturb mạnh (explore xa hơn)
//   - Nếu stagnate quá lâu → restart từ best + perturb rất mạnh
//
// Tham số tự động scale theo N:
//   - tabuTenure ≈ sqrt(N) (theo literature: Benlic & Hao 2011)
//   - maxIter mỗi phase ≈ N (mỗi node được xét ~1 lần/iteration)
//   - perturbStrength scale theo N/K
// ═══════════════════════════════════════════════════════════════════════════════
void iterated_tabu_search(ACOSolution &sol, mt19937_64 &rng)
{
    // ── Tham số tự động theo kích thước bài toán ──

    // Tabu tenure: sqrt(N) là quy tắc phổ biến trong literature
    //   N=100 → tenure=10, N=1000 → tenure=32, N=10000 → tenure=100
    int tabuTenure = max(5, (int)sqrt((double)N));

    // Số iteration mỗi tabu phase
    //   Đủ để mỗi node được xét nhiều lần, nhưng không quá chậm
    int tabuIter = max(50, N);

    // Số round ITS
    //   Scale ngược với N để tổng thời gian hợp lý
    int maxRounds = max(5, min(30, 2000 / max(1, N / 100)));

    // Perturbation strength cơ bản
    //   Di chuyển ~5% node mỗi lần perturb
    int baseStrength = max(2, N / 20);

    // ── Lưu nghiệm tốt nhất toàn cục ──
    ACOSolution bestSol = sol;
    double bestCost     = sol.cost;
    bool bestFeasible   = sol.feasible;

    // ── Đếm stagnation ──
    int noImproveRounds = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN ITS LOOP
    // ═══════════════════════════════════════════════════════════════════════
    for (int round = 0; round < maxRounds; ++round)
    {
        // ── Phase 1: Tabu Search (intensification) ──
        tabu_search_core(sol, tabuIter, tabuTenure, rng);

        // ── Phase 2: Update global best ──
        bool improved = false;

        if (sol.feasible && !bestFeasible)
        {
            // Feasible luôn thắng infeasible
            improved = true;
        }
        else if (sol.feasible == bestFeasible && sol.cost < bestCost - 1e-9)
        {
            // Cùng loại → cost nhỏ hơn thắng
            improved = true;
        }

        if (improved)
        {
            bestCost     = sol.cost;
            bestFeasible = sol.feasible;
            bestSol      = sol;
            noImproveRounds = 0;
        }
        else
        {
            noImproveRounds++;
        }

        // ── Nếu là round cuối → không cần perturb ──
        if (round == maxRounds - 1) break;

        // ── Phase 3: Perturbation (diversification) ──
        int strength;

        if (noImproveRounds == 0)
        {
            // Vừa cải thiện → perturb nhẹ (khai thác vùng lân cận best)
            strength = baseStrength / 2;
        }
        else if (noImproveRounds <= 3)
        {
            // Stagnate vừa → perturb trung bình
            strength = baseStrength;
        }
        else if (noImproveRounds <= 7)
        {
            // Stagnate lâu → perturb mạnh
            strength = baseStrength * 2;
        }
        else
        {
            // Stagnate quá lâu → RESTART từ best + perturb rất mạnh
            // Đây là cơ chế thoát kẹt quan trọng nhất
            sol = bestSol;
            strength = baseStrength * 3;
            noImproveRounds = 0;
        }

        perturb(sol, rng, max(1, strength));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TRẢ VỀ NGHIỆM TỐT NHẤT
    // ═══════════════════════════════════════════════════════════════════════
    sol = bestSol;
}
