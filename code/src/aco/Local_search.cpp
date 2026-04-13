#include "ACO.h"
#include "Local_search.h"

// ═══════════════════════════════════════════════════════════════════════════
// HÀM LOCAL SEARCH: Cải thiện nghiệm clustering hiện tại
//
// Bài toán: Gán N node vào K cluster, mỗi cluster có ràng buộc trọng số
//           (lower/upper bound trên M_weights chiều), tối thiểu tổng
//           khoảng cách intra-cluster.
//
// Chiến lược:
//   1. RELOCATE: chuyển 1 node từ cluster này sang cluster khác
//   2. SWAP:     đổi chỗ 2 node ở 2 cluster khác nhau
//
// Tham số:
//   sol      — nghiệm hiện tại (sẽ bị thay đổi in-place)
//   rng      — bộ sinh số ngẫu nhiên (Mersenne Twister 64-bit)
//   maxMoves — giới hạn số lần thay đổi tối đa
// ═══════════════════════════════════════════════════════════════════════════

void local_search(ACOSolution &sol, mt19937_64 &rng, int maxMoves)
{
    // ─────────────────────────────────────────────────────────────────────
    // 0. GUARD: nếu không cho phép move nào → thoát ngay
    // ─────────────────────────────────────────────────────────────────────
    if (maxMoves <= 0) return;

    // ─────────────────────────────────────────────────────────────────────
    // 1. LẤY THAM CHIẾU ĐẾN DỮ LIỆU TRONG SOLUTION
    //    (tránh copy, thao tác trực tiếp trên nghiệm)
    // ─────────────────────────────────────────────────────────────────────

    // assign[i] = cluster mà node i đang thuộc (0-indexed)
    auto &assign = sol.assign;

    // members[k] = vector chứa danh sách node thuộc cluster k
    auto &members = sol.members;

    // clusterWt[k][w] = tổng trọng số chiều w của tất cả node trong cluster k
    //   Ví dụ: cluster 2 có node {0,3,5}, thì clusterWt[2][w] = Wmat[0][w] + Wmat[3][w] + Wmat[5][w]
    auto &clusterWt = sol.clusterWeight;

    // sumDist[i][k] = tổng khoảng cách từ node i đến MỌI node hiện tại trong cluster k
    //   Ví dụ: cluster 1 = {2,4,7}, thì sumDist[i][1] = dist(i,2) + dist(i,4) + dist(i,7)
    //   Dùng để tính nhanh delta distance khi relocate/swap
    auto &sumDist = sol.clusterSumDist;

    // ─────────────────────────────────────────────────────────────────────
    // 2. PRE-ALLOCATE BUFFER (tránh cấp phát heap trong vòng lặp nóng)
    //
    //    Mỗi lần thử 1 move, ta cần "giả lập" trọng số cluster sau move.
    //    Thay vì tạo vector<double> mới mỗi lần (chậm vì malloc),
    //    ta tạo sẵn buffer 1 lần rồi ghi đè giá trị.
    // ─────────────────────────────────────────────────────────────────────

    // Buffer cho RELOCATE: trọng số giả lập của cluster nguồn & đích
    vector<double> simFrom(M_weights);  // simulated weight cluster nguồn
    vector<double> simTo(M_weights);    // simulated weight cluster đích

    // Buffer cho SWAP: trọng số giả lập của 2 cluster khi đổi node
    vector<double> simA(M_weights);     // simulated weight cluster A
    vector<double> simB(M_weights);     // simulated weight cluster B

    // ─────────────────────────────────────────────────────────────────────
    // 3. CACHE VIOLATION CHO TỪNG CLUSTER
    //
    //    violation[k] = tổng mức vi phạm ràng buộc trọng số của cluster k
    //    = Σ max(0, clusterWt[k][w] - WUmat[k][w])       (vượt upper bound)
    //    + Σ max(0, WLmat[k][w] - clusterWt[k][w])       (dưới lower bound)
    //
    //    Cache lại để không phải tính lại mỗi lần evaluate move.
    //    Chỉ update khi cluster thực sự thay đổi.
    // ─────────────────────────────────────────────────────────────────────

    // Hàm tính violation từ vector trọng số (có thể là thật hoặc giả lập)
    // cluster: index của cluster (dùng để tra WUmat, WLmat)
    // wt:      vector trọng số (M_weights chiều)
    // return:  tổng lượng vi phạm (>= 0, == 0 nghĩa là feasible)
    auto computeViolation = [&](int cluster, const vector<double> &wt) -> double
    {
        double viol = 0.0;               // tích lũy tổng vi phạm

        for (int w = 0; w < M_weights; ++w) // duyệt từng chiều trọng số
        {
            // Kiểm tra vượt upper bound
            if (wt[w] > WUmat[cluster][w] + VALID_EPS)
                viol += wt[w] - WUmat[cluster][w];    // phần vượt quá

            // Kiểm tra dưới lower bound
            if (wt[w] < WLmat[cluster][w] - VALID_EPS)
                viol += WLmat[cluster][w] - wt[w];    // phần thiếu
        }

        return viol;
    };

    // Cache violation hiện tại cho mỗi cluster
    vector<double> violCache(K);                       // violCache[k] = violation của cluster k

    for (int k = 0; k < K; ++k)                       // khởi tạo cache
        violCache[k] = computeViolation(k, clusterWt[k]);

    // Tổng violation toàn cục (== 0 nghĩa là nghiệm feasible)
    double totalViol = 0.0;
    for (int k = 0; k < K; ++k)
        totalViol += violCache[k];

    // ─────────────────────────────────────────────────────────────────────
    // 4. HELPER FUNCTIONS
    // ─────────────────────────────────────────────────────────────────────

    // Xóa 1 node khỏi vector trong O(1): swap với phần tử cuối rồi pop
    // Không giữ thứ tự, nhưng ta không cần thứ tự trong members[]
    auto eraseFromVec = [](vector<int> &vec, int node)
    {
        // Tìm vị trí node trong vector
        auto it = find(vec.begin(), vec.end(), node);

        // Ghi đè bằng phần tử cuối
        *it = vec.back();

        // Xóa phần tử cuối (O(1))
        vec.pop_back();
    };

    // ─────────────────────────────────────────────────────────────────────
    // 5. BIẾN ĐIỀU KHIỂN VÒNG LẶP
    // ─────────────────────────────────────────────────────────────────────

    int moveCount = 0;                                  // số move đã thực hiện
    bool improved = true;                               // flag: vòng trước có cải thiện không

    // Danh sách node (sẽ shuffle mỗi vòng để duyệt ngẫu nhiên)
    vector<int> nodeOrder(N);
    iota(nodeOrder.begin(), nodeOrder.end(), 0);        // fill [0, 1, 2, ..., N-1]

    // Bộ random chọn node (cho swap sampling)
    uniform_int_distribution<int> randNode(0, N - 1);

    // Số lần sample swap ngẫu nhiên cho mỗi node
    // min(100, N): đủ để có xác suất cao tìm được swap tốt
    const int SWAP_SAMPLES = min(100, N);

    // Epsilon so sánh: tránh chấp nhận thay đổi quá nhỏ (nhiễu số)
    constexpr double SCORE_EPS = 1e-6;

    // ═══════════════════════════════════════════════════════════════════════
    // 6. VÒNG LẶP LOCAL SEARCH CHÍNH
    //
    //    Mỗi iteration:
    //      a) Phân loại cluster (overloaded / underloaded / feasible)
    //      b) Thử RELOCATE
    //      c) Nếu relocate không improve → thử SWAP
    //      d) Lặp lại cho đến khi hết improve hoặc hết quota move
    // ═══════════════════════════════════════════════════════════════════════

    while (improved && moveCount < maxMoves)
    {
        improved = false;                               // reset flag đầu mỗi vòng

        // Shuffle để duyệt node theo thứ tự ngẫu nhiên → tránh bias
        shuffle(nodeOrder.begin(), nodeOrder.end(), rng);

        // ─────────────────────────────────────────────────────────────────
        // 6a. PHÂN LOẠI CLUSTER: overloaded / underloaded
        //
        //     overloaded:  cluster có ít nhất 1 chiều vượt upper bound
        //     underloaded: cluster có ít nhất 1 chiều dưới lower bound
        //
        //     Mục đích: relocate ưu tiên chuyển node TỪ overloaded
        //               SANG underloaded để giảm violation.
        // ─────────────────────────────────────────────────────────────────

        // Kiểm tra xem nghiệm đã feasible chưa
        bool isFeasible = (totalViol < SCORE_EPS);

        // Danh sách cluster nguồn (sẽ lấy node ra)
        vector<int> srcClusters;

        // Danh sách cluster đích (sẽ nhận node vào)
        vector<int> dstClusters;

        if (isFeasible)
        {
            // ĐÃ FEASIBLE → cho phép relocate giữa BẤT KỲ 2 cluster
            //   để tối ưu distance (nhưng phải giữ feasible)
            srcClusters.resize(K);
            dstClusters.resize(K);
            iota(srcClusters.begin(), srcClusters.end(), 0);   // [0,1,...,K-1]
            iota(dstClusters.begin(), dstClusters.end(), 0);
        }
        else
        {
            // CHƯA FEASIBLE → relocate có mục tiêu:
            //   nguồn = cluster overloaded
            //   đích  = cluster underloaded
            for (int c = 0; c < K; ++c)
            {
                bool overloaded  = false;        // có chiều nào vượt upper?
                bool underloaded = false;        // có chiều nào dưới lower?

                for (int w = 0; w < M_weights; ++w)
                {
                    if (clusterWt[c][w] > WUmat[c][w] + VALID_EPS)
                        overloaded = true;

                    if (clusterWt[c][w] < WLmat[c][w] - VALID_EPS)
                        underloaded = true;
                }

                if (overloaded)  srcClusters.push_back(c);
                if (underloaded) dstClusters.push_back(c);
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // 6b. RELOCATE: Chuyển 1 node từ srcCluster sang dstCluster
        //
        //     Cho mỗi node trong mỗi srcCluster:
        //       - Thử tất cả dstCluster
        //       - Chọn cluster đích cho score tốt nhất (best improvement)
        //       - score = (giảm violation) * PENALTY_SCALE - (tăng distance)
        //       - Nếu feasible: chỉ chấp nhận khi SAU move vẫn feasible
        //         VÀ distance giảm
        //       - Nếu score > 0 → thực hiện move
        // ─────────────────────────────────────────────────────────────────

        for (int fc : srcClusters)                      // fc = from cluster
        {
            int idx = 0;                                // index duyệt members[fc]

            // Duyệt từng node trong cluster nguồn
            while (idx < (int)members[fc].size() && moveCount < maxMoves)
            {
                int node = members[fc][idx];            // node đang xét

                // ── Tìm cluster đích tốt nhất ──

                int    bestDst   = -1;                  // cluster đích tốt nhất (-1 = chưa tìm)
                double bestScore = 0.0;                 // score tốt nhất (> 0 mới chấp nhận)

                for (int tc : dstClusters)              // tc = to cluster
                {
                    if (tc == fc) continue;             // không chuyển về chính nó

                    // ── Giả lập trọng số sau khi move ──
                    for (int w = 0; w < M_weights; ++w)
                    {
                        // Cluster nguồn: mất đi trọng số của node
                        simFrom[w] = clusterWt[fc][w] - Wmat[node][w];

                        // Cluster đích: nhận thêm trọng số của node
                        simTo[w]   = clusterWt[tc][w] + Wmat[node][w];
                    }

                    // ── Tính thay đổi violation ──
                    double violBefore = violCache[fc] + violCache[tc];
                                                        // violation hiện tại (dùng cache)

                    double violAfter  = computeViolation(fc, simFrom)
                                      + computeViolation(tc, simTo);
                                                        // violation sau move (tính từ sim)

                    // ── Tính thay đổi distance ──
                    // sumDist[node][tc] = tổng dist(node, mọi node trong tc)
                    // sumDist[node][fc] = tổng dist(node, mọi node trong fc)
                    // deltaDist > 0 nghĩa là distance tăng (xấu hơn)
                    double deltaDist = sumDist[node][tc] - sumDist[node][fc];

                    // ── Tính score tùy trạng thái ──
                    double score;

                    if (isFeasible)
                    {
                        // ĐÃ FEASIBLE: chỉ chấp nhận nếu:
                        //   1. Sau move VẪN feasible (violAfter ≈ 0)
                        //   2. Distance giảm (deltaDist < 0)
                        if (violAfter < SCORE_EPS && deltaDist < -SCORE_EPS)
                            score = -deltaDist;         // score = lượng distance giảm (dương)
                        else
                            score = -1.0;               // reject: không thỏa mãn
                    }
                    else
                    {
                        // CHƯA FEASIBLE: ưu tiên giảm violation, có trừ delta distance
                        // score = (violation giảm) * hệ số penalty - (distance tăng)
                        score = (violBefore - violAfter) * PENALTY_SCALE - deltaDist;
                    }

                    // Cập nhật best nếu score cao hơn
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestDst   = tc;
                    }
                }
                // ── Kết thúc duyệt dstClusters cho node này ──

                // ── Thực hiện move nếu tìm được cải thiện ──
                if (bestDst >= 0)
                {
                    int fc_old = fc;                    // cluster nguồn (alias cho rõ)
                    int tc_new = bestDst;               // cluster đích

                    // ── Cập nhật assign ──
                    assign[node] = tc_new;              // node giờ thuộc cluster mới

                    // ── Cập nhật members ──
                    //    Xóa node khỏi cluster cũ bằng swap-with-back (O(1))
                    members[fc_old][idx] = members[fc_old].back();
                    members[fc_old].pop_back();
                    //    ⚠️ KHÔNG tăng idx vì phần tử mới đã nằm ở vị trí idx

                    //    Thêm node vào cluster mới
                    members[tc_new].push_back(node);

                    // ── Cập nhật trọng số cluster ──
                    for (int w = 0; w < M_weights; ++w)
                    {
                        clusterWt[fc_old][w] -= Wmat[node][w]; // cluster cũ giảm
                        clusterWt[tc_new][w] += Wmat[node][w]; // cluster mới tăng
                    }

                    // ── Cập nhật violation cache ──
                    //    Chỉ 2 cluster bị ảnh hưởng
                    double oldViol_fc = violCache[fc_old];
                    double oldViol_tc = violCache[tc_new];

                    violCache[fc_old] = computeViolation(fc_old, clusterWt[fc_old]);
                    violCache[tc_new] = computeViolation(tc_new, clusterWt[tc_new]);

                    // Cập nhật totalViol
                    totalViol += (violCache[fc_old] - oldViol_fc)
                               + (violCache[tc_new] - oldViol_tc);

                    // ── Cập nhật sumDist ──
                    //
                    //    sumDist[v][k] = Σ dist(v, m) cho m ∈ members[k]
                    //
                    //    Khi node rời fc_old và vào tc_new:
                    //      - Với mọi node v ≠ node:
                    //          sumDist[v][fc_old] giảm dist(v, node)
                    //          sumDist[v][tc_new] tăng  dist(v, node)
                    //      - Với chính node: cần rebuild 2 cluster bị ảnh hưởng

                    // Phần 1: cập nhật cho mọi node khác (O(N))
                    for (int v = 0; v < N; ++v)
                    {
                        if (v == node) continue;        // node tự xử riêng

                        double d = distmat[v][node];    // khoảng cách v ↔ node

                        sumDist[v][fc_old] -= d;        // fc_old mất node
                        sumDist[v][tc_new] += d;        // tc_new có thêm node
                    }

                    // Phần 2: rebuild sumDist cho chính node (chỉ 2 cluster)
                    //
                    //    CHỈ fc_old và tc_new thay đổi member,
                    //    các cluster khác không đổi → sumDist[node][k] giữ nguyên
                    //
                    //    Complexity: O(|fc_old| + |tc_new|) ≈ O(N/K)
                    //    (so với O(N*K) nếu rebuild toàn bộ)

                    {
                        double s = 0.0;                 // tính sumDist[node][fc_old]
                        for (int m : members[fc_old])   // duyệt member MỚI của fc_old
                            s += distmat[node][m];
                        sumDist[node][fc_old] = s;
                    }

                    {
                        double s = 0.0;                 // tính sumDist[node][tc_new]
                        for (int m : members[tc_new])   // duyệt member MỚI của tc_new
                            s += distmat[node][m];
                        // Lưu ý: members[tc_new] bao gồm cả node →
                        //   distmat[node][node] = 0, nên không ảnh hưởng giá trị
                        sumDist[node][tc_new] = s;
                    }

                    // ── Đánh dấu có cải thiện, tăng counter ──
                    moveCount++;
                    improved = true;

                    // idx KHÔNG tăng (vector đã shift, phần tử mới ở idx)
                }
                else
                {
                    idx++;                              // không move → sang node tiếp
                }
            }
            // ── Kết thúc duyệt members[fc] ──
        }
        // ── Kết thúc duyệt srcClusters ──

        // Nếu relocate đã cải thiện → quay lại đầu while (re-classify cluster)
        if (improved) continue;

        // ─────────────────────────────────────────────────────────────────
        // 6c. SWAP: Đổi chỗ 2 node ở 2 cluster khác nhau
        //
        //     Khi relocate không cải thiện được, thử swap:
        //       - Duyệt nodeA theo thứ tự ngẫu nhiên
        //       - Với mỗi nodeA, sample ngẫu nhiên SWAP_SAMPLES nodeB
        //       - Nếu cùng cluster → skip
        //       - Tính delta tổng hợp = deltaDist + deltaPenalty
        //       - Nếu deltaTotal < 0 → chấp nhận (first improvement)
        //       - Break ngay khi tìm được swap tốt → quay lại while
        // ─────────────────────────────────────────────────────────────────

        for (int ii = 0; ii < N && moveCount < maxMoves; ++ii)
        {
            int nodeA    = nodeOrder[ii];               // node A (ngẫu nhiên theo shuffle)
            int clusterA = assign[nodeA];               // cluster chứa A

            for (int t = 0; t < SWAP_SAMPLES && moveCount < maxMoves; ++t)
            {
                int nodeB = randNode(rng);              // chọn ngẫu nhiên node B

                if (nodeA == nodeB) continue;           // trùng node → skip

                int clusterB = assign[nodeB];           // cluster chứa B

                if (clusterA == clusterB) continue;     // cùng cluster → swap vô nghĩa

                // ── Tính delta distance khi swap A ↔ B ──
                //
                //    Trước swap:
                //      cost_A = sumDist[A][clusterA]   (dist A tới mọi node cùng cluster A)
                //      cost_B = sumDist[B][clusterB]   (dist B tới mọi node cùng cluster B)
                //
                //    Sau swap:
                //      cost_A' = sumDist[A][clusterB]  (A giờ ở B, dist tới mọi node cluster B)
                //      cost_B' = sumDist[B][clusterA]  (B giờ ở A, dist tới mọi node cluster A)
                //
                //    NHƯNG: sumDist[A][clusterB] bao gồm dist(A,B) (vì B ∈ clusterB)
                //           Sau swap, B không còn ở clusterB → phải trừ dist(A,B)
                //           Tương tự cho sumDist[B][clusterA] với A
                //           Và sau swap, A ∈ clusterB nên phải cộng dist(A, chính A) = 0
                //           Tương tự B ∈ clusterA.
                //
                //    Delta = (cost_A' + cost_B') - (cost_A + cost_B)
                //          = (sumDist[A][clB] - dist(A,B))       — A tới clB trừ B
                //          + (sumDist[B][clA] - dist(B,A))       — B tới clA trừ A
                //          - sumDist[A][clA] + dist(A,A)         — trước: A tới clA trừ chính A
                //          - sumDist[B][clB] + dist(B,B)         — trước: B tới clB trừ chính B
                //
                //    Vì dist(A,A) = dist(B,B) = 0, dist(A,B) = dist(B,A):
                //
                //    Delta = (sumDist[A][clB] - sumDist[A][clA])
                //          + (sumDist[B][clA] - sumDist[B][clB])
                //          - 2 * dist(A, B)

                double deltaDist =
                    (sumDist[nodeA][clusterB] - sumDist[nodeA][clusterA])  // A: clA → clB
                  + (sumDist[nodeB][clusterA] - sumDist[nodeB][clusterB]) // B: clB → clA
                  - 2.0 * distmat[nodeA][nodeB];                          // correction term

                // ── Giả lập trọng số cluster sau swap ──
                for (int w = 0; w < M_weights; ++w)
                {
                    // Cluster A: mất nodeA, nhận nodeB
                    simA[w] = clusterWt[clusterA][w]
                            - Wmat[nodeA][w]
                            + Wmat[nodeB][w];

                    // Cluster B: mất nodeB, nhận nodeA
                    simB[w] = clusterWt[clusterB][w]
                            - Wmat[nodeB][w]
                            + Wmat[nodeA][w];
                }

                // ── Tính delta violation ──
                double violBefore = violCache[clusterA] + violCache[clusterB];
                                                        // violation hiện tại (cache)

                double violAfter  = computeViolation(clusterA, simA)
                                  + computeViolation(clusterB, simB);
                                                        // violation sau swap (simulated)

                // ── Tính tổng delta cost ──
                //    deltaTotal < 0 → cải thiện
                //    = deltaDist + (violAfter - violBefore) * PENALTY_SCALE
                double deltaTotal = deltaDist
                                  + (violAfter - violBefore) * PENALTY_SCALE;

                // ── Chấp nhận swap nếu cải thiện ──
                if (deltaTotal < -SCORE_EPS)
                {
                    // ── Cập nhật assign ──
                    assign[nodeA] = clusterB;           // A giờ thuộc cluster B
                    assign[nodeB] = clusterA;           // B giờ thuộc cluster A

                    // ── Cập nhật members (O(1) mỗi operation) ──
                    eraseFromVec(members[clusterA], nodeA);  // xóa A khỏi clA
                    eraseFromVec(members[clusterB], nodeB);  // xóa B khỏi clB

                    members[clusterA].push_back(nodeB);      // thêm B vào clA
                    members[clusterB].push_back(nodeA);      // thêm A vào clB

                    // ── Cập nhật trọng số cluster (đã tính sẵn) ──
                    clusterWt[clusterA] = simA;         // swap bằng buffer đã tính
                    clusterWt[clusterB] = simB;
                    // ⚠️ simA, simB là buffer pre-alloc, gán vào clusterWt là COPY
                    //    (chấp nhận được vì M_weights nhỏ, thường < 10)

                    // ── Cập nhật violation cache ──
                    double oldViol_A = violCache[clusterA];
                    double oldViol_B = violCache[clusterB];

                    violCache[clusterA] = computeViolation(clusterA, clusterWt[clusterA]);
                    violCache[clusterB] = computeViolation(clusterB, clusterWt[clusterB]);

                    totalViol += (violCache[clusterA] - oldViol_A)
                               + (violCache[clusterB] - oldViol_B);

                    // ── Cập nhật sumDist ──
                    //
                    //    Khi swap A ↔ B (A rời clA vào clB, B rời clB vào clA):
                    //
                    //    Với mọi node v ≠ A, v ≠ B:
                    //      sumDist[v][clA] += dist(v,B) - dist(v,A)
                    //                         (clA mất A, nhận B)
                    //      sumDist[v][clB] += dist(v,A) - dist(v,B)
                    //                         (clB mất B, nhận A)
                    //
                    //    Với A và B: rebuild 2 cluster bị ảnh hưởng

                    // Phần 1: cập nhật cho mọi node khác (O(N))
                    for (int v = 0; v < N; ++v)
                    {
                        if (v == nodeA || v == nodeB) continue;

                        double dA = distmat[v][nodeA];  // dist(v, A)
                        double dB = distmat[v][nodeB];  // dist(v, B)

                        sumDist[v][clusterA] += dB - dA; // clA: -A +B
                        sumDist[v][clusterB] += dA - dB; // clB: -B +A
                    }

                    // Phần 2: rebuild sumDist cho nodeA và nodeB
                    //         CHỈ 2 cluster bị ảnh hưởng (clA và clB)
                    //
                    //    Complexity: O(|clA| + |clB|) ≈ O(N/K) cho mỗi node
                    //    Tổng: O(2 * N/K) cho cả 2 node

                    for (int k : {clusterA, clusterB})  // chỉ 2 cluster thay đổi
                    {
                        double sA = 0.0;                // sumDist[nodeA][k]
                        double sB = 0.0;                // sumDist[nodeB][k]

                        for (int m : members[k])        // duyệt member mới của k
                        {
                            sA += distmat[nodeA][m];    // dist(A, m)
                            sB += distmat[nodeB][m];    // dist(B, m)
                        }
                        // Lưu ý: nếu nodeA ∈ members[k] → dist(A,A)=0 → OK
                        //         nếu nodeB ∈ members[k] → dist(B,B)=0 → OK

                        sumDist[nodeA][k] = sA;
                        sumDist[nodeB][k] = sB;
                    }

                    // ── Đánh dấu cải thiện ──
                    moveCount++;
                    improved = true;

                    break;                              // first improvement → thoát inner loop
                }
            }
            // ── Kết thúc SWAP_SAMPLES cho nodeA ──

            // Nếu đã tìm được swap → thoát outer loop, quay lại while
            if (improved) break;
        }
        // ── Kết thúc duyệt nodeOrder (swap) ──

    }
    // ═══════════════════════════════════════════════════════════════════════
    // KẾT THÚC LOCAL SEARCH
    //
    // sol đã được cập nhật in-place:
    //   - sol.assign:          cluster assignment mới
    //   - sol.members:         danh sách member mới
    //   - sol.clusterWeight:   trọng số cluster mới
    //   - sol.clusterSumDist:  sumDist mới
    // ═══════════════════════════════════════════════════════════════════════
}
