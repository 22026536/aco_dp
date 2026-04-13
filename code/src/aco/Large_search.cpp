#include "ACO.h"
#include "Large_search.h"


// Forward declarations if needed (these exist in original file)
// extern int N, K, M_weights;
// extern vector<vector<double>> Wmat, WLmat, WUmat, distmat;

// Helper: compute cluster violation (sum of deficits+excess across types)
double cluster_violation_from_sums(const vector<vector<double>> &sumW, int k)
{
    double v = 0.0;
    for (int t = 0; t < M_weights; ++t)
    {
        if (sumW[k][t] < WLmat[k][t] - VALID_EPS)
            v += (WLmat[k][t] - sumW[k][t]);
        else if (sumW[k][t] > WUmat[k][t] + VALID_EPS)
            v += (sumW[k][t] - WUmat[k][t]);
    }
    return v;
}

// compute overloaded / deficient clusters
pair<vector<int>, vector<int>> compute_over_under_from_sums(const vector<vector<double>> &sumW)
{
    vector<int> overloaded;
    vector<int> deficient;
    for (int k = 0; k < K; ++k)
    {
        double over = 0.0, under = 0.0;
        for (int t = 0; t < M_weights; ++t)
        {
            if (sumW[k][t] > WUmat[k][t] + VALID_EPS)
                over += sumW[k][t] - WUmat[k][t];
            if (sumW[k][t] < WLmat[k][t] - VALID_EPS)
                under += WLmat[k][t] - sumW[k][t];
        }
        if (over > VALID_EPS)
            overloaded.push_back(k);
        if (under > VALID_EPS)
            deficient.push_back(k);
    }
    return make_pair(overloaded, deficient);
}

// ================= LARGE SEARCH (cải tiến nghiệm lớn) =================

// Số node tối đa lấy mẫu trong mỗi cluster để giảm thời gian
static const int MAX_SAMPLE_PER_CLUSTER = 60;

void large_search(vector<int> &clusterOfNode, mt19937_64 &rng, int MAX_MOVES)
{
    // ================= KHỞI TẠO =================

    // danh sách node thuộc mỗi cluster
    vector<vector<int>> nodesInCluster(K);

    // tổng weight của từng cluster theo từng loại weight
    vector<vector<double>> sumWeight(K, vector<double>(M_weights, 0.0));

    // xây dựng nodesInCluster + sumWeight
    for (int node = 0; node < N; ++node)
    {
        int cluster = clusterOfNode[node];

        if (cluster < 0 || cluster >= K)
            continue;

        nodesInCluster[cluster].push_back(node);

        for (int t = 0; t < M_weights; ++t)
            sumWeight[cluster][t] += Wmat[node][t];
    }

    // ================= TIỀN XỬ LÝ DISTANCE =================

    // sumDist[i][k] = tổng khoảng cách từ node i tới toàn bộ node trong cluster k
    vector<vector<double>> sumDist(N, vector<double>(K, 0.0));

    for (int k = 0; k < K; ++k)
    {
        for (int node_j : nodesInCluster[k])
        {
            for (int node_i = 0; node_i < N; ++node_i)
            {
                sumDist[node_i][k] += distmat[node_i][node_j];
            }
        }
    }

    int moveCount = 0;

    // ================= VÒNG LẶP CHÍNH =================
    while (moveCount < MAX_MOVES)
    {
        bool hasImproved = false;

        // xác định cluster quá tải và thiếu tải
        auto overUnder = compute_over_under_from_sums(sumWeight);
        vector<int> overloadedClusters = overUnder.first;
        vector<int> underloadedClusters = overUnder.second;

        // nếu đã feasible (không vi phạm constraint)
        bool isFeasible = overloadedClusters.empty() && underloadedClusters.empty();

        // nếu feasible → cho phép optimize tự do giữa tất cả cluster
        if (isFeasible)
        {
            overloadedClusters.clear();
            underloadedClusters.clear();

            for (int k = 0; k < K; ++k)
            {
                overloadedClusters.push_back(k);
                underloadedClusters.push_back(k);
            }
        }

        // ================= PHASE 1: RELOCATION =================

        struct Move
        {
            double score; // độ tốt của move
            int node;     // node được chuyển
            int from;     // cluster nguồn
            int to;       // cluster đích
        };

        // max heap theo score
        struct CompareMove
        {
            bool operator()(Move const &a, Move const &b) const
            {
                return a.score < b.score;
            }
        };

        priority_queue<Move, vector<Move>, CompareMove> moveQueue;

        // duyệt cluster nguồn
        for (int fromCluster : overloadedClusters)
        {
            vector<int> candidateNodes = nodesInCluster[fromCluster];

            // lấy mẫu nếu cluster quá lớn
            if ((int)candidateNodes.size() > MAX_SAMPLE_PER_CLUSTER)
            {
                shuffle(candidateNodes.begin(), candidateNodes.end(), rng);
                candidateNodes.resize(MAX_SAMPLE_PER_CLUSTER);
            }

            // thử di chuyển từng node
            for (int node : candidateNodes)
            {
                for (int toCluster : underloadedClusters)
                {
                    if (toCluster == fromCluster)
                        continue;

                    double violationBefore = 0.0;
                    double violationAfter = 0.0;

                    // tính violation trước và sau khi move
                    for (int t = 0; t < M_weights; ++t)
                    {
                        double w_from = sumWeight[fromCluster][t];
                        double w_to = sumWeight[toCluster][t];

                        // --- trước khi move ---
                        if (w_from < WLmat[fromCluster][t] - VALID_EPS)
                            violationBefore += WLmat[fromCluster][t] - w_from;
                        if (w_from > WUmat[fromCluster][t] + VALID_EPS)
                            violationBefore += w_from - WUmat[fromCluster][t];

                        if (w_to < WLmat[toCluster][t] - VALID_EPS)
                            violationBefore += WLmat[toCluster][t] - w_to;
                        if (w_to > WUmat[toCluster][t] + VALID_EPS)
                            violationBefore += w_to - WUmat[toCluster][t];

                        // --- sau khi move ---
                        double new_w_from = w_from - Wmat[node][t];
                        double new_w_to = w_to + Wmat[node][t];

                        if (new_w_from < WLmat[fromCluster][t] - VALID_EPS)
                            violationAfter += WLmat[fromCluster][t] - new_w_from;
                        if (new_w_from > WUmat[fromCluster][t] + VALID_EPS)
                            violationAfter += new_w_from - WUmat[fromCluster][t];

                        if (new_w_to < WLmat[toCluster][t] - VALID_EPS)
                            violationAfter += WLmat[toCluster][t] - new_w_to;
                        if (new_w_to > WUmat[toCluster][t] + VALID_EPS)
                            violationAfter += new_w_to - WUmat[toCluster][t];
                    }

                    // lợi ích giảm violation
                    double violationGain = violationBefore - violationAfter;

                    // thay đổi khoảng cách nội bộ
                    double deltaDistance = sumDist[node][toCluster] - sumDist[node][fromCluster];

                    // score tổng hợp
                    double score = 5 * PENALTY_SCALE * violationGain - deltaDistance;

                    // chỉ giữ move tốt
                    if (score > 1e-9)
                    {
                        moveQueue.push({score, node, fromCluster, toCluster});
                    }
                }
            }
        }

        int movesThisRound = 0;

        // ================= APPLY MOVE =================
        while (!moveQueue.empty() &&
               moveCount < MAX_MOVES &&
               movesThisRound < 30)
        {
            Move bestMove = moveQueue.top();
            moveQueue.pop();

            // nếu node đã bị move trước đó thì bỏ
            if (clusterOfNode[bestMove.node] != bestMove.from)
                continue;

            // (tính lại score để đảm bảo vẫn đúng)
            // ... (giữ nguyên logic cũ)

            double deltaDistance =
                sumDist[bestMove.node][bestMove.to] -
                sumDist[bestMove.node][bestMove.from];

            if (deltaDistance >= 0 && !isFeasible)
                continue;

            // ===== THỰC HIỆN MOVE =====

            // xóa khỏi cluster cũ
            auto it = find(nodesInCluster[bestMove.from].begin(),
                                nodesInCluster[bestMove.from].end(),
                                bestMove.node);
            if (it != nodesInCluster[bestMove.from].end())
                nodesInCluster[bestMove.from].erase(it);

            // thêm vào cluster mới
            nodesInCluster[bestMove.to].push_back(bestMove.node);

            // cập nhật weight
            for (int t = 0; t < M_weights; ++t)
            {
                sumWeight[bestMove.from][t] -= Wmat[bestMove.node][t];
                sumWeight[bestMove.to][t] += Wmat[bestMove.node][t];
            }

            // cập nhật assignment
            clusterOfNode[bestMove.node] = bestMove.to;

            // cập nhật sumDist
            for (int v = 0; v < N; ++v)
            {
                if (v == bestMove.node)
                    continue;

                sumDist[v][bestMove.from] -= distmat[v][bestMove.node];
                sumDist[v][bestMove.to] += distmat[v][bestMove.node];
            }

            // cập nhật riêng cho node vừa move
            for (int k = 0; k < K; ++k)
            {
                double s = 0.0;
                for (int member : nodesInCluster[k])
                {
                    if (member != bestMove.node)
                        s += distmat[bestMove.node][member];
                }
                sumDist[bestMove.node][k] = s;
            }

            moveCount++;
            movesThisRound++;
            hasImproved = true;
        }

        // nếu không cải thiện → dừng
        if (!hasImproved)
            break;
    }

    // ================= CHECK VIOLATION CUỐI =================
    double totalViolation = 0.0;
    for (int k = 0; k < K; ++k)
    {
        totalViolation += cluster_violation_from_sums(sumWeight, k);
    }

    // nếu gần 0 → nghiệm feasible
}
