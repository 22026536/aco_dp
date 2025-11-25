#include "ACO.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <cassert>
#include <vector>
#include <numeric>
#include <functional>
#include <chrono>
#include <fstream>
#include <string>
#include <queue>
#include <iomanip>
#include <sstream>
#include <locale>

using std::mt19937_64;
using std::vector;
using Clock = chrono::steady_clock;

// ================== GLOBALS ==================
int N = 0, K = 0, M_weights = 0;
static std::vector<int> log_iter;
static std::vector<double> log_time; // elapsed time at snapshot
struct LogRow
{
    int iter;
    double time;
    double bestCost;
    bool bestFeasible;
    double bestThisIter;
    int feasibleAnts;
    int noImprove;
};
static std::vector<LogRow> log_rows;

extern Parameters parameters;

static std::string LOG_EVOL_FILENAME;
static std::string LOG_COST_FILENAME;
static std::string LOG_SOLU_FILENAME;

// New unified weight matrices (you already have these)
vector<vector<double>> Wmat;  // Wmat[i][t] : weight t of vertex i
vector<vector<double>> WLmat; // WLmat[k][t] : lower bound cluster k attribute t
vector<vector<double>> WUmat; // WUmat[k][t] : upper bound cluster k attribute t

// ===== Backwards-compatible globals (restore for existing code) =====
// Many places in your code still reference these old names.
// Define them here and keep them synchronized with the matrices above.
vector<double> w1, w2;                     // per-node weights (first two dims)
vector<vector<double>> distmat;            // distance matrix
vector<double> Wmin1, Wmax1, Wmin2, Wmax2; // per-cluster bounds for first two dims

double PENALTY_SCALE = 10000.0;
double OVERLOAD_PENALTY_FACTOR = 0.5;
bool ALLOW_VIOLATIONS = true;
const double VALID_EPS = 1e-6;

// Format: fixed notation (no exponent), choose decimals (e.g. 0 => integer)
static inline std::string format_cost_fixed(double v, int decimals = 0)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << v;
    return oss.str();
}

// Format: with thousands separators (comma or locale-specific).
// Note: std::locale("") uses system locale; if not set, it may not insert separators.
// This returns integer format if decimals==0, otherwise with decimals.
static inline std::string format_cost_with_commas(double v, int decimals = 0)
{
    std::ostringstream oss;
    try
    {
        oss.imbue(std::locale("")); // system locale, may enable grouping
    }
    catch (...)
    {
        // ignore if locale not supported
    }
    oss << std::fixed << std::setprecision(decimals) << v;
    return oss.str();
}

// Kiểm tra tính hợp lệ của trọng số trong instance
bool check_weights_validity(const Instance instance)
{
    int N = instance.nV;
    int K = instance.nK;
    int T = instance.nT;

    if (N <= 0 || K <= 0 || T <= 0)
    {
        cerr << "[ERROR] Invalid sizes\n";
        return false;
    }

    vector<double> sumNode(T, 0.0);
    vector<double> sumMin(T, 0.0);
    vector<double> sumMax(T, 0.0);

    // Tổng trọng số node
    for (int i = 0; i < N; ++i)
    {
        for (int t = 0; t < T; ++t)
            sumNode[t] += instance.W[i][t];
    }

    // Tổng min-max cụm
    for (int k = 0; k < K; ++k)
    {
        for (int t = 0; t < T; ++t)
        {
            sumMin[t] += instance.WL[k][t];
            sumMax[t] += instance.WU[k][t];
        }
    }

    // Kiểm tra từng loại trọng số
    for (int t = 0; t < T; ++t)
    {
        if (sumNode[t] < sumMin[t] - 1e-9 || sumNode[t] > sumMax[t] + 1e-9)
        {
            cerr << "\n[INVALID] Weight type " << t << "\n";
            cerr << "  Sum node = " << sumNode[t] << "\n";
            cerr << "  Sum min  = " << sumMin[t] << "\n";
            cerr << "  Sum max  = " << sumMax[t] << "\n";
            return false;
        }
    }

    return true;
}

// Replace existing compute_cost with this unified version
double compute_cost(const vector<int> &assign)
{
    // 1) intra distance
    double intra = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (assign[i] == assign[j])
                intra += distmat[i][j] + distmat[j][i];

    // 2) invalid cluster id
    for (int i = 0; i < N; ++i)
        if (assign[i] < 0 || assign[i] >= K)
            return 1e300;

    // 3) sum weights
    vector<vector<double>> sumW(K, vector<double>(M_weights, 0.0));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            sumW[assign[i]][t] += Wmat[i][t];

    // 4) tổng VI PHẠM của tất cả trọng số
    double total_violation = 0.0;

    for (int k = 0; k < K; ++k)
    {
        for (int t = 0; t < M_weights; ++t)
        {
            double s = sumW[k][t];
            double low = WLmat[k][t];
            double high = WUmat[k][t];

            if (s < low)
                total_violation += (low - s);
            if (s > high)
                total_violation += (s - high);
        }
    }

    // 5) phạt = tổng vi phạm * PENALTY_SCALE
    double penalty = total_violation * PENALTY_SCALE;

    return intra + penalty;
}

bool is_feasible(const std::vector<int> &assign)
{
    // Basic size checks
    if ((int)assign.size() != N)
        return false;

    // Ensure Wmat/WLmat/WUmat are initialized and have correct dims
    if ((int)Wmat.size() != N || (int)WLmat.size() != K || (int)WUmat.size() != K)
        return false;

    // If no weight dimensions, accept only if M_weights == 0
    if (M_weights == 0)
    {
        // still need to ensure Wmat rows exist (or treat as zero-dim)
        return true;
    }

    // defensive: Wmat must have at least one row and WLmat/WUmat must have at least one column
    if (Wmat.empty() || WLmat.empty() || WUmat.empty())
        return false;
    if ((int)Wmat[0].size() != M_weights || (int)WLmat[0].size() != M_weights || (int)WUmat[0].size() != M_weights)
        return false;

    // Build clusters and check bounds of cluster ids
    std::vector<std::vector<int>> sol(K);
    std::vector<char> included(N, 0);

    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        if (c < 0 || c >= K)
            return false;
        if (included[i]) // defensive (shouldn't happen since we iterate once), keep for parity with original
            return false;
        included[i] = 1;
        sol[c].push_back(i);
    }

    // Ensure every vertex is included (defensive)
    if (std::any_of(included.begin(), included.end(), [](char v)
                    { return v == 0; }))
        return false;

    // Check weight constraints for ALL attributes t = 0..M_weights-1

    for (int k = 0; k < K; ++k)
    {
        for (int t = 0; t < M_weights; ++t)
        {
            double wkt = 0.0;
            for (int v : sol[k])
                wkt += Wmat[v][t];

            // scale according to Solution::Validate pattern to obtain relative tolerance
            double scale = std::max(1.0, std::max(std::abs(wkt), std::max(std::abs(WLmat[k][t]), std::abs(WUmat[k][t]))));
            double tol = VALID_EPS * scale;

            if (wkt + tol < WLmat[k][t] || wkt - tol > WUmat[k][t])
            {
                // violation of lower or upper bound (beyond tolerance)
                return false;
            }
        }
    }

    // All checks passed
    return true;
}

// ----------------------- repair_solution (with MCF subproblem) -----------------------
// ---------- LIGHTWEIGHT local_search (same as before) ----------
void local_search(vector<int> &assign, mt19937_64 &rng, int maxMoves)
{
    if (maxMoves <= 0)
        return;

    // ---------------------------------------
    // 0. Build members & sumW (multi-weights)
    // ---------------------------------------
    vector<vector<int>> members(K);
    vector<vector<double>> sumW(K, vector<double>(M_weights, 0.0));

    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        members[c].push_back(i);
        for (int t = 0; t < M_weights; ++t)
            sumW[c][t] += Wmat[i][t];
    }

    // ---------------------------------------
    // 1. sumDist[i][k] = sum dist(i, members[k])
    // ---------------------------------------
    vector<vector<double>> sumDist(N, vector<double>(K, 0.0));
    for (int k = 0; k < K; ++k)
        for (int j : members[k])
            for (int i = 0; i < N; ++i)
                sumDist[i][k] += distmat[i][j];

    // Shuffle nodes to avoid deterministic behavior
    vector<int> nodes(N);
    iota(nodes.begin(), nodes.end(), 0);
    shuffle(nodes.begin(), nodes.end(), rng);

    int moves = 0;
    bool improved = true;

    const int sampleSwapK = min(30, max(10, N / 4));
    uniform_int_distribution<int> uniNode(0, N - 1);

    // Helper: compute violation sum for cluster k
    auto cluster_violation = [&](int k, const vector<double> &sw)
    {
        double over = 0, under = 0;
        for (int t = 0; t < M_weights; ++t)
        {
            if (sw[t] > WUmat[k][t] + VALID_EPS)
                over += sw[t] - WUmat[k][t];
            if (sw[t] < WLmat[k][t] - VALID_EPS)
                under += WLmat[k][t] - sw[t];
        }
        return over + under;
    };

    while (improved && moves < maxMoves)
    {
        improved = false;

        // ---------------------------------------------------------
        // 2. FIND OVERLOADED / DEFICIENT CLUSTERS (multi-weights)
        // ---------------------------------------------------------
        vector<int> overloaded, deficient;

        for (int k = 0; k < K; ++k)
        {
            double over = 0, under = 0;
            for (int t = 0; t < M_weights; ++t)
            {
                if (sumW[k][t] > WUmat[k][t] + VALID_EPS)
                    over += sumW[k][t] - WUmat[k][t];
                if (sumW[k][t] < WLmat[k][t] - VALID_EPS)
                    under += WLmat[k][t] - sumW[k][t];
            }
            if (over > 1e-9)
                overloaded.push_back(k);
            if (under > 1e-9)
                deficient.push_back(k);
        }

        // ---------------------------------------------------------
        // 3. MULTI-DIMENSION RELOCATE FOR OVERLOADED → DEFICIENT
        // ---------------------------------------------------------
        for (int k_from : overloaded)
        {
            for (int u_idx = 0; u_idx < (int)members[k_from].size() && moves < maxMoves; ++u_idx)
            {
                int u = members[k_from][u_idx];

                int bestTo = k_from;
                double bestGain = 0.0;

                for (int k_to : deficient)
                {
                    if (k_to == k_from)
                        continue;

                    // Compute violation before move
                    double viol_before =
                        cluster_violation(k_from, sumW[k_from]) +
                        cluster_violation(k_to, sumW[k_to]);

                    // Compute hypothetical sums after move
                    vector<double> new_from = sumW[k_from];
                    vector<double> new_to = sumW[k_to];
                    for (int t = 0; t < M_weights; ++t)
                    {
                        new_from[t] -= Wmat[u][t];
                        new_to[t] += Wmat[u][t];
                    }

                    double viol_after =
                        cluster_violation(k_from, new_from) +
                        cluster_violation(k_to, new_to);

                    double gain = viol_before - viol_after;
                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestTo = k_to;
                    }
                }

                if (bestTo != k_from)
                {
                    // ------------------
                    // APPLY the relocate
                    // ------------------
                    auto it = find(members[k_from].begin(), members[k_from].end(), u);
                    if (it != members[k_from].end())
                        members[k_from].erase(it);

                    members[bestTo].push_back(u);

                    // Update sumW
                    for (int t = 0; t < M_weights; ++t)
                    {
                        sumW[k_from][t] -= Wmat[u][t];
                        sumW[bestTo][t] += Wmat[u][t];
                    }

                    assign[u] = bestTo;

                    // Update sumDist incrementally
                    for (int v = 0; v < N; ++v)
                    {
                        if (v == u)
                            continue;
                        sumDist[v][k_from] -= distmat[v][u];
                        sumDist[v][bestTo] += distmat[v][u];
                    }

                    // Recompute row u
                    for (int kk = 0; kk < K; ++kk)
                    {
                        double s = 0;
                        for (int j : members[kk])
                            s += distmat[u][j];
                        sumDist[u][kk] = s;
                    }

                    moves++;
                    improved = true;
                }
            }
        }

        if (improved)
            continue;

        // ---------------------------------------------------------
        // 4. ENHANCED SWAP (MULTI-WEIGHTS)
        // ---------------------------------------------------------
        for (int ii = 0; ii < N && moves < maxMoves; ++ii)
        {
            int i = nodes[ii];
            int ci = assign[i];

            for (int trial = 0; trial < sampleSwapK && moves < maxMoves; ++trial)
            {
                int j = uniNode(rng);
                if (j == i)
                    continue;

                int cj = assign[j];
                if (ci == cj)
                    continue;

                // Compute delta intra-distance
                double deltaSwap =
                    (sumDist[i][cj] - sumDist[i][ci]) +
                    (sumDist[j][ci] - sumDist[j][cj]) -
                    2.0 * distmat[i][j];

                // Compute penalty change for multi-weights
                double penBefore = cluster_violation(ci, sumW[ci]) +
                                   cluster_violation(cj, sumW[cj]);

                vector<double> ns_ci = sumW[ci];
                vector<double> ns_cj = sumW[cj];

                for (int t = 0; t < M_weights; ++t)
                {
                    ns_ci[t] = sumW[ci][t] - Wmat[i][t] + Wmat[j][t];
                    ns_cj[t] = sumW[cj][t] - Wmat[j][t] + Wmat[i][t];
                }

                double penAfter = cluster_violation(ci, ns_ci) +
                                  cluster_violation(cj, ns_cj);

                double deltaPen = (penAfter - penBefore) * PENALTY_SCALE;
                double deltaTotal = deltaSwap + deltaPen;

                if (deltaTotal < -1e-6)
                {
                    // ------------------
                    // APPLY SWAP
                    // ------------------
                    auto iti = find(members[ci].begin(), members[ci].end(), i);
                    if (iti != members[ci].end())
                        members[ci].erase(iti);
                    members[ci].push_back(j);

                    auto itj = find(members[cj].begin(), members[cj].end(), j);
                    if (itj != members[cj].end())
                        members[cj].erase(itj);
                    members[cj].push_back(i);

                    sumW[ci] = ns_ci;
                    sumW[cj] = ns_cj;
                    assign[i] = cj;
                    assign[j] = ci;

                    // Update sumDist incrementally
                    for (int v = 0; v < N; ++v)
                    {
                        if (v == i || v == j)
                            continue;
                        sumDist[v][ci] += distmat[v][j] - distmat[v][i];
                        sumDist[v][cj] += distmat[v][i] - distmat[v][j];
                    }

                    // Rebuild rows i, j
                    for (int kk = 0; kk < K; ++kk)
                    {
                        double si = 0, sj = 0;
                        for (int m : members[kk])
                        {
                            si += distmat[i][m];
                            sj += distmat[j][m];
                        }
                        sumDist[i][kk] = si;
                        sumDist[j][kk] = sj;
                    }

                    moves++;
                    improved = true;
                    break;
                }
            }
            if (improved)
                break;
        }

    } // end while
}

// Tunable parameters
static const double REPAIR_VIOL_WEIGHT = 1e6; // large to prefer reducing violations over intra-cost
static const double REPAIR_DIST_WEIGHT = 1.0; // weight for intra-distance in score
static const int REPAIR_MAX_MOVES = 2000;
static const int REPAIR_SAMPLE_PER_CLUSTER = 60;
static const int REPAIR_MAX_SWAPS = 800;

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
std::pair<vector<int>, vector<int>> compute_over_under_from_sums(const vector<vector<double>> &sumW)
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
    return std::make_pair(overloaded, deficient);
}

// Multi-node greedy relocation to cover deficits (atomic moves)
void multi_relocate_for_deficit(vector<int> &assign,
                                vector<vector<int>> &members,
                                vector<vector<double>> &sumW,
                                vector<vector<double>> &sumDist,
                                mt19937_64 &rng)
{
    // compute per-cluster deficits/excess
    vector<vector<double>> deficit(K, vector<double>(M_weights, 0.0));
    vector<vector<double>> excess(K, vector<double>(M_weights, 0.0));
    vector<double> totalDeficit(K, 0.0), totalExcess(K, 0.0);

    for (int k = 0; k < K; ++k)
    {
        for (int t = 0; t < M_weights; ++t)
        {
            if (sumW[k][t] < WLmat[k][t] - VALID_EPS)
            {
                deficit[k][t] = WLmat[k][t] - sumW[k][t];
                totalDeficit[k] += deficit[k][t];
            }
            else if (sumW[k][t] > WUmat[k][t] + VALID_EPS)
            {
                excess[k][t] = sumW[k][t] - WUmat[k][t];
                totalExcess[k] += excess[k][t];
            }
        }
    }

    vector<int> overloaded;
    for (int k = 0; k < K; ++k)
        if (totalExcess[k] > VALID_EPS)
            overloaded.push_back(k);
    if (overloaded.empty())
        return;

    // For each deficient cluster, pick nodes greedily
    for (int to = 0; to < K; ++to)
    {
        if (totalDeficit[to] <= VALID_EPS)
            continue;

        struct Cand
        {
            int node;
            int from;
            double cover;
            double cost;
            double score;
        };
        vector<Cand> candidates;
        candidates.reserve(1024);

        for (int from : overloaded)
        {
            for (int u : members[from])
            {
                double cover = 0.0;
                for (int t = 0; t < M_weights; ++t)
                {
                    double need = deficit[to][t];
                    if (need <= VALID_EPS)
                        continue;
                    double contribute = std::min(need, Wmat[u][t]);
                    cover += contribute;
                }
                if (cover <= VALID_EPS)
                    continue;
                double deltaIntra = sumDist[u][to] - sumDist[u][from];
                double cost = deltaIntra;
                double score = cost / cover; // smaller is better
                candidates.push_back(Cand{u, from, cover, cost, score});
            }
        }

        if (candidates.empty())
            continue;
        std::sort(candidates.begin(), candidates.end(), [](const Cand &a, const Cand &b)
                  {
            if (a.score == b.score) return a.cover > b.cover;
            return a.score < b.score; });

        double covered = 0.0;
        vector<Cand> chosen;
        vector<char> node_selected(N, 0);

        for (auto &c : candidates)
        {
            if (covered >= totalDeficit[to] - VALID_EPS)
                break;
            if (node_selected[c.node])
                continue;
            chosen.push_back(c);
            node_selected[c.node] = 1;
            covered += c.cover;
        }

        if (covered + 1e-9 < totalDeficit[to])
            continue; // couldn't cover

        // Apply chosen moves atomically
        for (auto &c : chosen)
        {
            int u = c.node;
            int from = c.from;
            auto it = std::find(members[from].begin(), members[from].end(), u);
            if (it != members[from].end())
                members[from].erase(it);
            members[to].push_back(u);
            for (int t = 0; t < M_weights; ++t)
            {
                sumW[from][t] -= Wmat[u][t];
                sumW[to][t] += Wmat[u][t];
            }
            assign[u] = to;
            for (int v = 0; v < N; ++v)
            {
                if (v == u)
                    continue;
                sumDist[v][from] -= distmat[v][u];
                sumDist[v][to] += distmat[v][u];
            }
            for (int kk = 0; kk < K; ++kk)
            {
                double s = 0.0;
                for (int member : members[kk])
                    if (member != u)
                        s += distmat[u][member];
                sumDist[u][kk] = s;
            }
        }

        // update deficits for 'to' (not used further in this simple implementation)
    }
}

// Main replacement for repair_solution (C++14-compatible)
void repair_solution(std::vector<int> &assign, std::mt19937_64 &rng)
{
    // Build members and sumW
    vector<vector<int>> members(K);
    vector<vector<double>> sumW(K, vector<double>(M_weights, 0.0));
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        if (c < 0 || c >= K)
            continue;
        members[c].push_back(i);
        for (int t = 0; t < M_weights; ++t)
            sumW[c][t] += Wmat[i][t];
    }

    // precompute sumDist
    vector<vector<double>> sumDist(N, vector<double>(K, 0.0));
    for (int k = 0; k < K; ++k)
    {
        for (int j : members[k])
        {
            for (int i = 0; i < N; ++i)
                sumDist[i][k] += distmat[i][j];
        }
    }

    int moves = 0;
    bool didSomething = true;

    while (didSomething && moves < REPAIR_MAX_MOVES)
    {
        didSomething = false;

        std::pair<vector<int>, vector<int>> tmp_pair = compute_over_under_from_sums(sumW);
        vector<int> overloaded = tmp_pair.first;
        vector<int> deficient = tmp_pair.second;

        if (overloaded.empty() && deficient.empty())
            break;

        // Priority queue of candidate relocations
        struct Move
        {
            double score;
            int u, from, to;
        };
        struct Cmp
        {
            bool operator()(Move const &a, Move const &b) const { return a.score < b.score; }
        };
        std::priority_queue<Move, vector<Move>, Cmp> pq;

        for (size_t idx_from = 0; idx_from < overloaded.size(); ++idx_from)
        {
            int from = overloaded[idx_from];
            vector<int> candNodes = members[from];
            if ((int)candNodes.size() > REPAIR_SAMPLE_PER_CLUSTER)
            {
                std::shuffle(candNodes.begin(), candNodes.end(), rng);
                candNodes.resize(REPAIR_SAMPLE_PER_CLUSTER);
            }
            for (size_t ii = 0; ii < candNodes.size(); ++ii)
            {
                int u = candNodes[ii];
                for (size_t idx_to = 0; idx_to < deficient.size(); ++idx_to)
                {
                    int to = deficient[idx_to];
                    if (to == from)
                        continue;

                    double beforePair = 0.0, afterPair = 0.0;
                    for (int t = 0; t < M_weights; ++t)
                    {
                        double s_from = sumW[from][t];
                        double s_to = sumW[to][t];

                        double before_from = 0.0, before_to = 0.0;
                        if (s_from < WLmat[from][t] - VALID_EPS)
                            before_from += WLmat[from][t] - s_from;
                        if (s_from > WUmat[from][t] + VALID_EPS)
                            before_from += s_from - WUmat[from][t];
                        if (s_to < WLmat[to][t] - VALID_EPS)
                            before_to += WLmat[to][t] - s_to;
                        if (s_to > WUmat[to][t] + VALID_EPS)
                            before_to += s_to - WUmat[to][t];
                        beforePair += before_from + before_to;

                        double ns_from = s_from - Wmat[u][t];
                        double ns_to = s_to + Wmat[u][t];
                        double after_from = 0.0, after_to = 0.0;
                        if (ns_from < WLmat[from][t] - VALID_EPS)
                            after_from += WLmat[from][t] - ns_from;
                        if (ns_from > WUmat[from][t] + VALID_EPS)
                            after_from += ns_from - WUmat[from][t];
                        if (ns_to < WLmat[to][t] - VALID_EPS)
                            after_to += WLmat[to][t] - ns_to;
                        if (ns_to > WUmat[to][t] + VALID_EPS)
                            after_to += ns_to - WUmat[to][t];
                        afterPair += after_from + after_to;
                    }

                    double violGain = beforePair - afterPair;
                    double deltaIntra = (sumDist[u][to] - sumDist[u][from]);
                    double score = REPAIR_VIOL_WEIGHT * violGain - REPAIR_DIST_WEIGHT * deltaIntra;
                    if (score > 1e-9)
                        pq.push(Move{score, u, from, to});
                }
            }
        }

        int appliedThisRound = 0;
        while (!pq.empty() && moves < REPAIR_MAX_MOVES && appliedThisRound < 30)
        {
            Move mv = pq.top();
            pq.pop();
            if (assign[mv.u] != mv.from)
                continue;

            double beforePair = 0.0, afterPair = 0.0;
            for (int t = 0; t < M_weights; ++t)
            {
                double s_from = sumW[mv.from][t];
                double s_to = sumW[mv.to][t];

                if (s_from < WLmat[mv.from][t] - VALID_EPS)
                    beforePair += WLmat[mv.from][t] - s_from;
                if (s_from > WUmat[mv.from][t] + VALID_EPS)
                    beforePair += s_from - WUmat[mv.from][t];
                if (s_to < WLmat[mv.to][t] - VALID_EPS)
                    beforePair += WLmat[mv.to][t] - s_to;
                if (s_to > WUmat[mv.to][t] + VALID_EPS)
                    beforePair += s_to - WUmat[mv.to][t];

                double ns_from = s_from - Wmat[mv.u][t];
                double ns_to = s_to + Wmat[mv.u][t];
                if (ns_from < WLmat[mv.from][t] - VALID_EPS)
                    afterPair += WLmat[mv.from][t] - ns_from;
                if (ns_from > WUmat[mv.from][t] + VALID_EPS)
                    afterPair += ns_from - WUmat[mv.from][t];
                if (ns_to < WLmat[mv.to][t] - VALID_EPS)
                    afterPair += WLmat[mv.to][t] - ns_to;
                if (ns_to > WUmat[mv.to][t] + VALID_EPS)
                    afterPair += ns_to - WUmat[mv.to][t];
            }

            double violGain = beforePair - afterPair;
            if (violGain <= 1e-9)
                continue;

            double deltaIntra = (sumDist[mv.u][mv.to] - sumDist[mv.u][mv.from]);
            double score = REPAIR_VIOL_WEIGHT * violGain - REPAIR_DIST_WEIGHT * deltaIntra;
            if (score <= 1e-9)
                continue;

            auto it = std::find(members[mv.from].begin(), members[mv.from].end(), mv.u);
            if (it != members[mv.from].end())
                members[mv.from].erase(it);
            members[mv.to].push_back(mv.u);
            for (int t = 0; t < M_weights; ++t)
            {
                sumW[mv.from][t] -= Wmat[mv.u][t];
                sumW[mv.to][t] += Wmat[mv.u][t];
            }
            assign[mv.u] = mv.to;

            for (int v = 0; v < N; ++v)
            {
                if (v == mv.u)
                    continue;
                sumDist[v][mv.from] -= distmat[v][mv.u];
                sumDist[v][mv.to] += distmat[v][mv.u];
            }
            for (int kk = 0; kk < K; ++kk)
            {
                double s = 0.0;
                for (int member : members[kk])
                    if (member != mv.u)
                        s += distmat[mv.u][member];
                sumDist[mv.u][kk] = s;
            }

            moves++;
            appliedThisRound++;
            didSomething = true;
        }

        // If relocations did not resolve, try multi-node relocation
        if (!didSomething)
        {
            multi_relocate_for_deficit(assign, members, sumW, sumDist, rng);
            // after multi relocation, recompute overloaded/deficient in next loop
            // also try sampled swaps
            int swapsTried = 0;
            vector<int> nodes(N);
            for (int i = 0; i < N; ++i)
                nodes[i] = i;
            std::shuffle(nodes.begin(), nodes.end(), rng);

            for (int idx = 0; idx < N && swapsTried < REPAIR_MAX_SWAPS; ++idx)
            {
                int i = nodes[idx];
                int ci = assign[i];
                for (int trial = 0; trial < 8 && swapsTried < REPAIR_MAX_SWAPS; ++trial)
                {
                    int j = (int)(rng() % (uint64_t)N);
                    if (j == i)
                        continue;
                    int cj = assign[j];
                    if (ci == cj)
                        continue;

                    double before = 0.0, after = 0.0;
                    for (int t = 0; t < M_weights; ++t)
                    {
                        double s_ci = sumW[ci][t], s_cj = sumW[cj][t];
                        if (s_ci < WLmat[ci][t] - VALID_EPS)
                            before += WLmat[ci][t] - s_ci;
                        if (s_ci > WUmat[ci][t] + VALID_EPS)
                            before += s_ci - WUmat[ci][t];
                        if (s_cj < WLmat[cj][t] - VALID_EPS)
                            before += WLmat[cj][t] - s_cj;
                        if (s_cj > WUmat[cj][t] + VALID_EPS)
                            before += s_cj - WUmat[cj][t];

                        double ns_ci = s_ci - Wmat[i][t] + Wmat[j][t];
                        double ns_cj = s_cj - Wmat[j][t] + Wmat[i][t];
                        if (ns_ci < WLmat[ci][t] - VALID_EPS)
                            after += WLmat[ci][t] - ns_ci;
                        if (ns_ci > WUmat[ci][t] + VALID_EPS)
                            after += ns_ci - WUmat[ci][t];
                        if (ns_cj < WLmat[cj][t] - VALID_EPS)
                            after += WLmat[cj][t] - ns_cj;
                        if (ns_cj > WUmat[cj][t] + VALID_EPS)
                            after += ns_cj - WUmat[cj][t];
                    }
                    double violGain = before - after;
                    swapsTried++;
                    if (violGain <= 1e-9)
                        continue;

                    double deltaSwap = (sumDist[i][cj] - sumDist[i][ci]) + (sumDist[j][ci] - sumDist[j][cj]) - 2.0 * distmat[i][j];
                    double score = REPAIR_VIOL_WEIGHT * violGain - REPAIR_DIST_WEIGHT * deltaSwap;
                    if (score > 1e-9)
                    {
                        auto iti = std::find(members[ci].begin(), members[ci].end(), i);
                        if (iti != members[ci].end())
                            *iti = j;
                        auto itj = std::find(members[cj].begin(), members[cj].end(), j);
                        if (itj != members[cj].end())
                            *itj = i;
                        for (int t = 0; t < M_weights; ++t)
                        {
                            sumW[ci][t] = sumW[ci][t] - Wmat[i][t] + Wmat[j][t];
                            sumW[cj][t] = sumW[cj][t] - Wmat[j][t] + Wmat[i][t];
                        }
                        assign[i] = cj;
                        assign[j] = ci;
                        for (int v = 0; v < N; ++v)
                        {
                            if (v == i || v == j)
                                continue;
                            sumDist[v][ci] += distmat[v][j] - distmat[v][i];
                            sumDist[v][cj] += distmat[v][i] - distmat[v][j];
                        }
                        for (int kk = 0; kk < K; ++kk)
                        {
                            double si = 0.0, sj = 0.0;
                            for (int member : members[kk])
                            {
                                if (member == i)
                                    si += distmat[j][member];
                                else if (member == j)
                                    sj += distmat[i][member];
                                else
                                {
                                    si += distmat[i][member];
                                    sj += distmat[j][member];
                                }
                            }
                            sumDist[i][kk] = si;
                            sumDist[j][kk] = sj;
                        }
                        moves++;
                        didSomething = true;
                        break;
                    }
                }
                if (didSomething)
                    break;
            }
        }
    }

    // final: small-tolerance check to accept near-feasible
    double total_violation = 0.0;
    for (int k = 0; k < K; ++k)
        total_violation += cluster_violation_from_sums(sumW, k);
    if (total_violation <= 1e-6)
    {
        // accept as feasible within tolerance
        // (the calling code can check feasibility more strictly if desired)
    }
}

// ====================== MULTI-PASS REPAIR + LOCAL SEARCH ======================
void improve_ant_solution(vector<int> &assign, mt19937_64 &rng, int repairPasses, int localPasses)
{
    for (int r = 0; r < repairPasses; ++r)
    {
        repair_solution(assign, rng); // cân bằng overload + underload
    }

    for (int l = 0; l < localPasses; ++l)
    {
        local_search(assign, rng, 1000); // tối ưu chi phí giữ feasibility
    }

    // cuối cùng một lượt repair để đảm bảo feasibility trước khi cập nhật pheromone
    repair_solution(assign, rng); // cân bằng overload + underload
}

void SaveLogs(const ACOSolution &best)
{
    // 1) save evolution snapshots
    std::ofstream foutEvo(LOG_EVOL_FILENAME);
    if (!foutEvo.is_open())
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_EVOL_FILENAME << " for writing.\n";
    }
    else
    {
        // header
        foutEvo << "# iter   time(s)    bestCost    bestFeasible  bestThisIter  feasibleAnts  noImprove\n";
        for (auto &r : log_rows)
        {
            foutEvo << std::setw(6) << r.iter
                    << std::setw(12) << std::fixed << std::setprecision(4) << r.time
                    << std::setw(14) << std::fixed << std::setprecision(6) << r.bestCost
                    << std::setw(12) << (r.bestFeasible ? "1" : "0")
                    << std::setw(14) << std::fixed << std::setprecision(6) << r.bestThisIter
                    << std::setw(12) << r.feasibleAnts
                    << std::setw(12) << r.noImprove
                    << "\n";
        }
        foutEvo.close();
        std::cerr << "[SAVELOGS] evolution saved to " << LOG_EVOL_FILENAME << "\n";
    }

    // 2) save best cost (single value)
    std::ofstream foutCost(LOG_COST_FILENAME);
    if (!foutCost.is_open())
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_COST_FILENAME << " for writing.\n";
    }
    else
    {
        foutCost << std::fixed << std::setprecision(6) << best.cost << "\n";
        foutCost.close();
        std::cerr << "[SAVELOGS] best cost saved to " << LOG_COST_FILENAME << "\n";
    }

    // 3) save best solution (clusters)
    std::ofstream foutSolu(LOG_SOLU_FILENAME);
    if (!foutSolu.is_open())
    {
        std::cerr << "[SAVELOGS] Cannot open " << LOG_SOLU_FILENAME << " for writing.\n";
    }
    else
    {
        // best.assign is vector<int> of size N with cluster ids (0..K-1)
        std::vector<std::vector<int>> clusters(K);
        for (int i = 0; i < N; ++i)
        {
            int c = (i < (int)best.assign.size()) ? best.assign[i] : -1;
            if (c >= 0 && c < K)
                clusters[c].push_back(i + 1); // 1-based
        }
        for (int k = 0; k < K; ++k)
        {
            std::sort(clusters[k].begin(), clusters[k].end());
            for (int v : clusters[k])
                foutSolu << v << " ";
            foutSolu << "\n";
        }
        foutSolu.close();
        std::cerr << "[SAVELOGS] best solution saved to " << LOG_SOLU_FILENAME << "\n";
    }
}

ACOSolution ACO_tuned(const Instance &instance, int maxIter, double timeLimitSeconds, const string &instance_name)
{
    std::string base = parameters.LOGdir;
    if (base.empty())
        base = "results/logs/aco_logs";
    if (base.back() == '/')
        base.pop_back();

    LOG_EVOL_FILENAME = base + "/evolution/" + instance_name;
    LOG_COST_FILENAME = base + "/objectives/" + instance_name;
    LOG_SOLU_FILENAME = base + "/solutions/" + instance_name;
    // --- sanity checks ---
    if (!check_weights_validity(instance))
    {
        cerr << "[ERROR] Instance weight bounds inconsistent. Aborting ACO.\n";
        ACOSolution empty;
        empty.assign.clear();
        empty.cost = 1e300;
        empty.feasible = false;
        return empty;
    }

    // --- Initialize globals from instance ---
    N = instance.nV;
    K = instance.nK;
    M_weights = instance.nT;

    Wmat.assign(N, vector<double>(M_weights, 0.0));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            Wmat[i][t] = instance.W[i][t];

    WLmat.assign(K, vector<double>(M_weights, 0.0));
    WUmat.assign(K, vector<double>(M_weights, 0.0));
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < M_weights; ++t)
        {
            WLmat[k][t] = instance.WL[k][t];
            WUmat[k][t] = instance.WU[k][t];
        }

    distmat.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            distmat[i][j] = instance.D[i][j];

    w1.assign(N, 0.0);
    w2.assign(N, 0.0);
    for (int i = 0; i < N; ++i)
    {
        if (M_weights >= 1)
            w1[i] = Wmat[i][0];
        if (M_weights >= 2)
            w2[i] = Wmat[i][1];
    }

    Wmin1.assign(K, 0.0);
    Wmax1.assign(K, 0.0);
    Wmin2.assign(K, 0.0);
    Wmax2.assign(K, 0.0);
    for (int k = 0; k < K; ++k)
    {
        if (M_weights >= 1)
        {
            Wmin1[k] = WLmat[k][0];
            Wmax1[k] = WUmat[k][0];
        }
        if (M_weights >= 2)
        {
            Wmin2[k] = WLmat[k][1];
            Wmax2[k] = WUmat[k][1];
        }
    }

    if (K <= 0 || N <= 0)
    {
        cerr << "[ERROR] invalid N or K\n";
        ACOSolution empty;
        empty.assign.clear();
        empty.cost = 1e300;
        empty.feasible = false;
        return empty;
    }

    // --- ACO parameters (tunable) ---
    int m = min(N / 2, 40); // number of ants per iteration
    double alpha = 1.0;     // pheromone importance
    double beta = 2.0;      // desirability importance (larger => favor low delta cost)
    double rho = 0.1;       // evaporation
    double Q = 5000.0;      // pheromone deposit scale

    // selection temperature and q0 (small exploitation)
    double T_max = 1.0, T_min = 0.05;
    double Q0 = 0.05; // exploitation probability
    double Q_max = 1.0, Q_min = 0.05, Q_decay = 0.995;

    int L_candidates = min(K, 12);

    // repair configuration: choose topRepair ants (by pre-repair cost) to run local_search+repair
    int repairTop = 10; // you can set to m if you want all ants repaired

    mt19937_64 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> uni01(0.0, 1.0);

    // initial greedy solution (may be infeasible)
    ACOSolution best;

    // initialize pheromone matrix phi[i][k]
    vector<vector<double>> phi(N, vector<double>(K, 1.0));

    // prepare eta (distance-based) to build candidate lists quickly (cheap heuristic to limit K)
    vector<vector<double>> sumDist(N, vector<double>(K, 1.0));

    vector<vector<double>> eta(N, vector<double>(K, 0.0));
    auto update_eta = [&]()
    {
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                eta[i][k] = 1.0 / (1.0 + sumDist[i][k]);
    };

    // candidate lists per node (top-L by eta)
    vector<vector<int>> candidates(N);
    for (int i = 0; i < N; ++i)
    {
        vector<pair<double, int>> tmp;
        tmp.reserve(K);
        for (int k = 0; k < K; ++k)
            tmp.emplace_back(eta[i][k], k);
        sort(tmp.rbegin(), tmp.rend());
        int L = min((int)tmp.size(), L_candidates);
        candidates[i].clear();
        for (int x = 0; x < L; ++x)
            candidates[i].push_back(tmp[x].second);
        if (candidates[i].empty())
            candidates[i].push_back(0);
    }

    auto start = Clock::now();
    int iter = 0, noImprove = 0;

    // --- main loop ---
    while (iter < maxIter && chrono::duration<double>(Clock::now() - start).count() < timeLimitSeconds)
    {
        ++iter;
        double Q_iter = max(Q_min, Q_max * pow(Q_decay, (double)iter));

        vector<ACOSolution> ants(m);

        // construct each ant solution
        for (int a = 0; a < m; ++a)
        {
            ants[a].assign.assign(N, -1);
            // random node order
            // giữ track tổng trọng số cluster hiện tại (từ 0 đến K-1)
            vector<vector<double>> clusterWeight(K, vector<double>(M_weights, 0.0));
            // giữ track sumDist tương tự
            vector<vector<double>> clusterSumDist(K, vector<double>(N, 0.0)); // optional nếu dùng heuristic

            // random node order
            vector<int> nodes(N);
            iota(nodes.begin(), nodes.end(), 0);
            shuffle(nodes.begin(), nodes.end(), rng);

            for (int idx = 0; idx < N; ++idx)
            {
                int i = nodes[idx];
                const vector<int> &cand = candidates[i];
                double bestWeight = -1.0;
                int chosenK = cand[0];

                vector<double> weights(cand.size(), 0.0);

                for (int ci = 0; ci < (int)cand.size(); ++ci)
                {
                    int k = cand[ci];

                    // --- incremental cost estimation ---
                    // tính delta trọng số nếu gán node i vào cluster k
                    double penaltyDelta = 0.0;
                    for (int t = 0; t < M_weights; ++t)
                    {
                        double newSum = clusterWeight[k][t] + Wmat[i][t];
                        if (newSum > WUmat[k][t])
                            penaltyDelta += (newSum - WUmat[k][t]) * PENALTY_SCALE;
                    }

                    // tính heuristic distance incremental (sumDist)
                    double distHeur = clusterSumDist[k][i] + 1.0; // ví dụ: có thể + distmat[i][j] cho các j trong cluster

                    double desir = 1.0 / (1.0 + penaltyDelta + distHeur);
                    double tau = phi[i][k];
                    double weight = pow(tau, alpha) * pow(desir, beta);
                    weights[ci] = weight;

                    if (weight > bestWeight)
                    {
                        bestWeight = weight;
                        chosenK = k;
                    }
                }

                // --- selection (exploitation / roulette) ---
                double q = uni01(rng);
                if (q < Q0)
                {
                    // exploitation: dùng chosenK
                }
                else
                {
                    // roulette / softmax
                    double sumW = accumulate(weights.begin(), weights.end(), 0.0);
                    double pick = uni01(rng) * sumW;
                    double acc = 0.0;
                    for (int ci = 0; ci < (int)cand.size(); ++ci)
                    {
                        acc += weights[ci];
                        if (pick <= acc)
                        {
                            chosenK = cand[ci];
                            break;
                        }
                    }
                }

                ants[a].assign[i] = chosenK;

                // cập nhật incremental cluster weight
                for (int t = 0; t < M_weights; ++t)
                    clusterWeight[chosenK][t] += Wmat[i][t];

                // cập nhật sumDist nếu dùng
                for (int j = 0; j < N; ++j)
                    clusterSumDist[chosenK][j] += distmat[i][j];
            }

            // sau khi xây dựng xong ant, tính cost full 1 lần
            ants[a].cost = compute_cost(ants[a].assign);
            ants[a].feasible = is_feasible(ants[a].assign);
        } // ants built

        // sort ants by cost ascending (cost includes penalties) — prefer feasible implicitly by lower cost
        vector<int> order(m);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a1, int a2)
             { return ants[a1].cost < ants[a2].cost; });

        // pick top repairTop ants to local_search + repair
        for (int r = 0; r < min(repairTop, m); ++r)
        {
            int ai = order[r];
            // local_search and repair operate in-place
            improve_ant_solution(ants[ai].assign, rng, 1, 1);
            // recompute cost and feasibility
            ants[ai].cost = compute_cost(ants[ai].assign);
            ants[ai].feasible = is_feasible(ants[ai].assign);
        }

        // after repairs, resort by feasibility then cost
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a1, int a2)
             { return ants[a1].cost < ants[a2].cost; });

        // update global best (prefer feasible)
        bool improvedThisIter = false;
        for (int r = 0; r < m; ++r)
        {
            int ai = order[r];

            bool curFeasible = ants[ai].feasible;
            double curCost = ants[ai].cost;

            bool bestFeasible = best.feasible;
            double bestCost = best.cost;

            bool accept = false;

            if (curFeasible)
            {
                // (1) Nghiệm mới FEASIBLE:
                //    - Chấp nhận nếu best chưa feasible
                //    - Hoặc cost mới < cost hiện tại
                if (!bestFeasible || curCost + 1e-12 < bestCost)
                    accept = true;
            }
            else
            {
                // (2) Nghiệm mới INFEASIBLE:
                //    - Chỉ chấp nhận nếu best cũng INFEASIBLE
                //    - Và cost nhỏ hơn
                if (!bestFeasible && curCost + 1e-12 < bestCost)
                    accept = true;
            }

            if (accept)
            {
                best = ants[ai];
                improvedThisIter = true;
                noImprove = 0;

                auto now = Clock::now();
                double elapsed = chrono::duration<double>(now - start).count();
                cerr << "[ITER " << iter << "] New best cost=" << format_cost_with_commas(best.cost, 0)
                     << " (feasible=" << (best.feasible ? "YES" : "NO") << ", time " << elapsed << "s)\n";
            }
        }
        if (!improvedThisIter)
            ++noImprove;

        // update sumDist/eta occasionally from current best (so candidates adapt)
        if (best.assign.size() == (size_t)N && best.feasible)
        {
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < N; ++i)
                    sumDist[i][k] = 0.0;
            for (int j = 0; j < N; ++j)
            {
                int c = best.assign[j];
                if (c < 0 || c >= K)
                    continue;
                for (int i = 0; i < N; ++i)
                    sumDist[i][c] += distmat[i][j];
            }
            update_eta();
            if (iter % 20 == 0)
            {
                for (int i = 0; i < N; ++i)
                {
                    vector<pair<double, int>> tmp;
                    tmp.reserve(K);
                    for (int k = 0; k < K; ++k)
                        tmp.emplace_back(eta[i][k], k);
                    sort(tmp.rbegin(), tmp.rend());
                    int L = min((int)tmp.size(), L_candidates);
                    candidates[i].clear();
                    for (int x = 0; x < L; ++x)
                        candidates[i].push_back(tmp[x].second);
                    if (candidates[i].empty())
                        candidates[i].push_back(0);
                }
            }
        }

        // --- PHEROMONE UPDATE ---
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                phi[i][k] *= (1.0 - rho);

        // deposit Tmax for best ant only
        for (int i = 0; i < N; ++i)
        {
            int c = best.assign[i];
            if (c >= 0 && c < K)
                phi[i][c] += T_max;
        }

        // optionally, deposit Tmin for remaining ants
        for (int a = 0; a < m; ++a)
        {
            if (ants[a].assign == best.assign)
                continue; // skip best
            for (int i = 0; i < N; ++i)
            {
                int c = ants[a].assign[i];
                if (c >= 0 && c < K)
                    phi[i][c] += T_min;
            }
        }

        // clamp phi to avoid extremes
        const double PHI_MIN = 1e-6, PHI_MAX = 1e9;
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < K; ++k)
                phi[i][k] = max(PHI_MIN, min(PHI_MAX, phi[i][k]));

        // logging
        if (iter % 10 == 0)
        {
            double bestThis = 1e300;
            int feasCount = 0;
            for (int a = 0; a < m; ++a)
            {
                if (ants[a].feasible)
                {
                    feasCount++;
                    bestThis = min(bestThis, ants[a].cost);
                }
            }
            double elapsed = chrono::duration<double>(Clock::now() - start).count();
            cerr << "[ITER " << iter << "] bestGlobalCost=" << format_cost_with_commas(best.cost, 0)
                 << " (feasible=" << (best.feasible ? "YES" : "NO") << ")"
                 << " bestThisIter=" << bestThis
                 << " feasibleAnts=" << feasCount
                 << " noImprove=" << noImprove
                 << " (elapsed " << elapsed << "s)\n";
            // append snapshot to in-memory log
            LogRow r;
            r.iter = iter;
            r.time = elapsed;
            r.bestCost = best.cost;
            r.bestFeasible = best.feasible;
            r.bestThisIter = (bestThis < 1e299 ? bestThis : 1e300);
            r.feasibleAnts = feasCount;
            r.noImprove = noImprove;
            log_rows.push_back(r);
        }

        // stagnation reset
        int noImproveReset = 200;
        if (noImprove >= noImproveReset)
        {
            cerr << "[RESET] no improvement for" << noImproveReset << "-> reset pheromones\n";
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < K; ++k)
                    phi[i][k] = 1.0;
            noImprove = 0;
        }
    } // end while

    vector<vector<int>> clusters(K);
    for (int i = 0; i < N; ++i)
    {
        int c = (i < (int)best.assign.size()) ? best.assign[i] : -1;
        if (c >= 0 && c < K)
            clusters[c].push_back(i + 1);
    }
    for (int k = 0; k < K; ++k)
    {
        for (int node : clusters[k])
            cerr << node << " ";
        cerr << "\n";
    }

    if (!best.feasible)
        cout << "Final solution is invalid.\n";
    else
        cout << "Final solution is valid.\n";

    cout << "Final cost = " << format_cost_with_commas(best.cost, 0) << "\n";

    SaveLogs(best);

    return best;
}
