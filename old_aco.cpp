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

// ===== GLOBAL DEFINITIONS =====
vector<int> log_iter;
vector<double> log_time;
vector<LogRow> log_rows;

Parameters parameters;

string LOG_EVOL_FILENAME;
string LOG_COST_FILENAME;
string LOG_SOLU_FILENAME;

int N = 0;
int K = 0;
int M_weights = 0;

vector<vector<double>> Wmat;
vector<vector<double>> WLmat;
vector<vector<double>> WUmat;
vector<vector<double>> distmat;

double PENALTY_SCALE = 10000.0;
double VALID_EPS = 1e-6;

// Format: fixed notation (no exponent), choose decimals (e.g. 0 => integer)
static inline std::string format_cost_fixed(double v, int decimals)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << v;
    return oss.str();
}

// Format: with thousands separators (comma or locale-specific).
// Note: std::locale("") uses system locale; if not set, it may not insert separators.
// This returns integer format if decimals==0, otherwise with decimals.
static inline std::string format_cost_with_commas(double v, int decimals)
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

// local search
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

// large search
static const int SEARCH_SAMPLE_PER_CLUSTER = 60;
void large_search(std::vector<int> &assign, std::mt19937_64 &rng, int MAX_MOVES)
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

    while (moves < MAX_MOVES)
    {
        bool didSomething = false;

        std::pair<vector<int>, vector<int>> tmp_pair = compute_over_under_from_sums(sumW);
        vector<int> overloaded = tmp_pair.first;
        vector<int> deficient = tmp_pair.second;

        bool feasibleNow = overloaded.empty() && deficient.empty();

        // If already feasible → allow optimization among all clusters
        if (feasibleNow)
        {
            overloaded.clear();
            deficient.clear();
            for (int k = 0; k < K; ++k)
            {
                overloaded.push_back(k);
                deficient.push_back(k);
            }
        }

        // ------------------ Relocation phase ------------------
        struct Move
        {
            double score;
            int u, from, to;
        };
        struct Cmp
        {
            bool operator()(Move const &a, Move const &b) const
            {
                return a.score < b.score;
            }
        };
        std::priority_queue<Move, vector<Move>, Cmp> pq;

        for (int from : overloaded)
        {
            vector<int> candNodes = members[from];
            if ((int)candNodes.size() > SEARCH_SAMPLE_PER_CLUSTER)
            {
                std::shuffle(candNodes.begin(), candNodes.end(), rng);
                candNodes.resize(SEARCH_SAMPLE_PER_CLUSTER);
            }

            for (int u : candNodes)
            {
                for (int to : deficient)
                {
                    if (to == from)
                        continue;

                    double beforePair = 0.0, afterPair = 0.0;

                    for (int t = 0; t < M_weights; ++t)
                    {
                        double s_from = sumW[from][t];
                        double s_to = sumW[to][t];

                        // violation before
                        if (s_from < WLmat[from][t] - VALID_EPS)
                            beforePair += WLmat[from][t] - s_from;
                        if (s_from > WUmat[from][t] + VALID_EPS)
                            beforePair += s_from - WUmat[from][t];

                        if (s_to < WLmat[to][t] - VALID_EPS)
                            beforePair += WLmat[to][t] - s_to;
                        if (s_to > WUmat[to][t] + VALID_EPS)
                            beforePair += s_to - WUmat[to][t];

                        double ns_from = s_from - Wmat[u][t];
                        double ns_to = s_to + Wmat[u][t];

                        // violation after
                        if (ns_from < WLmat[from][t] - VALID_EPS)
                            afterPair += WLmat[from][t] - ns_from;
                        if (ns_from > WUmat[from][t] + VALID_EPS)
                            afterPair += ns_from - WUmat[from][t];

                        if (ns_to < WLmat[to][t] - VALID_EPS)
                            afterPair += WLmat[to][t] - ns_to;
                        if (ns_to > WUmat[to][t] + VALID_EPS)
                            afterPair += ns_to - WUmat[to][t];
                    }

                    double violGain = beforePair - afterPair;
                    double deltaIntra = sumDist[u][to] - sumDist[u][from];

                    double score = 5 * PENALTY_SCALE * violGain
                                   - deltaIntra;

                    if (score > 1e-9)
                        pq.push(Move{score, u, from, to});
                }
            }
        }

        int appliedThisRound = 0;

        while (!pq.empty() &&
               moves < MAX_MOVES &&
               appliedThisRound < 30)
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
            double deltaIntra = sumDist[mv.u][mv.to] - sumDist[mv.u][mv.from];

            double score = 5 * PENALTY_SCALE * violGain
                           - deltaIntra;

            if (score <= 1e-9)
                continue;

            // Apply move
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

        // If nothing improved → stop
        if (!didSomething)
            break;
    }

    // final tolerance check
    double total_violation = 0.0;
    for (int k = 0; k < K; ++k)
        total_violation += cluster_violation_from_sums(sumW, k);

    if (total_violation <= 1e-6)
    {
        // acceptable feasible
    }
}

int hamming_distance(const vector<int>& a, const vector<int>& b)
{
    int diff = 0;
    for (int i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) diff++;
    return diff;
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

    // --- Initialize globals from instance (MULTI-DIMENSION VERSION) ---

    N = instance.nV;                 // số node
    K = instance.nK;                 // số cụm
    M_weights = instance.nT;         // số chiều trọng số (resource dimensions)

    // ================= NODE WEIGHTS =================
    Wmat.assign(N, vector<double>(M_weights, 0.0));   // Wmat[i][t] = weight của node i tại chiều t
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            Wmat[i][t] = instance.W[i][t];            // copy từ instance


    // ================= CLUSTER CAPACITY BOUNDS =================
    WLmat.assign(K, vector<double>(M_weights, 0.0));  // lower bound mỗi cụm, mỗi chiều
    WUmat.assign(K, vector<double>(M_weights, 0.0));  // upper bound mỗi cụm, mỗi chiều

    for (int k = 0; k < K; ++k)
        for (int t = 0; t < M_weights; ++t)
        {
            WLmat[k][t] = instance.WL[k][t];          // min capacity cụm k tại chiều t
            WUmat[k][t] = instance.WU[k][t];          // max capacity cụm k tại chiều t
        }


    // ================= DISTANCE MATRIX =================
    distmat.assign(N, vector<double>(N, 0.0));        // ma trận khoảng cách giữa các node
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            distmat[i][j] = instance.D[i][j];


    // ================= VALIDITY CHECK =================
    if (K <= 0 || N <= 0 || M_weights <= 0)           // kiểm tra dữ liệu hợp lệ
    {
        cerr << "[ERROR] invalid N, K, or number of weight dimensions\n";
        ACOSolution empty;
        empty.assign.clear();
        empty.cost = 1e300;
        empty.feasible = false;
        return empty;
    }

    // tính pelnaty scale
    double BASE_COST = 0.0;
    int pairCount = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
        {
            BASE_COST += distmat[i][j];
            pairCount++;
        }

    double meanDist = BASE_COST / pairCount;

    // ===== AUTO DIST SCALE (instance-dependent) =====

    // kích thước cụm trung bình khi xây nghiệm
    double avgClusterSize = max(1.0, (double)N / K);

    // distHeur ~ sum distance from node i to cluster k
    // kỳ vọng ≈ avgClusterSize * meanDist
    double DIST_SCALE = avgClusterSize * meanDist;
    
    // tính pelnalty scale
    double sumW = 0.0;
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < M_weights; ++t)
            sumW += Wmat[i][t];

    double meanWeight = sumW / (N * M_weights);

    PENALTY_SCALE = 50.0 * (meanDist / meanWeight);
    // --- ACO parameters (tunable) ---
    int m = min(N / 2, 40); // number of ants per iteration
    double alpha = 1.25;    // pheromone importance
    double beta = 1.0;      // desirability importance (larger => favor low delta cost)
    double rho = 0.2;       // evaporation

    // selection temperature and q0 (small exploitation)
    double T_max = 0.2, T_min = 0.01;
    double Q_max = 0.8, Q_min = 0.1;
    double Q0 = Q_max;
    int STAGNATE_DROP = 0.05; // mỗi iteration stagnate, giảm Q0 0.05
    int STAGNATE_LIMIT = 10;

    // repair configuration: choose topRepair ants (by pre-repair cost) to run local_search+repair
    int repairTop = 5;

    mt19937_64 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> uni01(0.0, 1.0);

    // initial greedy solution (may be infeasible)
    ACOSolution best;

    best.assign.assign(N, 0);  // tất cả vào cluster 0
    best.cost = compute_cost(best.assign);
    best.feasible = is_feasible(best.assign);

    // initialize pheromone matrix phi[i][j]  (node-node)
    vector<vector<double>> phi(N, vector<double>(N, T_min));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            phi[i][j] = (i == j ? 0.0 : T_min);

    auto start = Clock::now();
    int iter = 0, noImprove = 0;

    // --- main loop ---
    while (iter < maxIter && chrono::duration<double>(Clock::now() - start).count() < timeLimitSeconds)
    {
        ++iter;
        vector<ACOSolution> ants(m);

        // construct each ant solution
        for (int a = 0; a < m; ++a)
        {
            ants[a].assign.assign(N, -1);
            // giữ track tổng trọng số cluster hiện tại (từ 0 đến K-1)
            vector<vector<double>> clusterWeight(K, vector<double>(M_weights, 0.0));
            // lưu delta increment
            vector<vector<double>> clusterSumDist(K, vector<double>(N, 0.0));
            // lưu pheromone increment
            vector<vector<double>> clusterSumPhero(K, vector<double>(N, 0.0));

            // random node order
            vector<int> nodes(N);
            iota(nodes.begin(), nodes.end(), 0);
            shuffle(nodes.begin(), nodes.end(), rng);

            int seedCount = min(K, N);

            // gán 1 node đầu cho mỗi cluster
            for (int k = 0; k < seedCount; ++k)
            {
                int i = nodes[k];

                ants[a].assign[i] = k;

                // cập nhật weight
                for (int t = 0; t < M_weights; ++t)
                    clusterWeight[k][t] += Wmat[i][t];

                // cập nhật dist
                for (int j = 0; j < N; ++j)
                    clusterSumDist[k][j] += distmat[i][j];

                // cập nhật pheromone
                for (int j = 0; j < N; ++j)
                    clusterSumPhero[k][j] += phi[i][j];
            }

            for (int idx = seedCount; idx < N; ++idx)
            {
                int i = nodes[idx];
                double bestWeight = -1.0;
                int chosenK = 0;
                vector<double> weights(K, 0.0);

                for (int k = 0; k < K; ++k)
                {
                    double dot = 0.0;
                    double normNeed = 0.0;
                    double normNode = 0.0;
                    double emptiness = 0.0;
                    double overflow = 0.0;

                    for (int t = 0; t < M_weights; ++t)
                    {
                        double need = max(0.0, WLmat[k][t] - clusterWeight[k][t]);
                        double w = Wmat[i][t];

                        double after = clusterWeight[k][t] + w;

                            if (after > WUmat[k][t])
                            {
                                double excess = (after - WUmat[k][t]) / WUmat[k][t];
                                overflow += excess;
                            }

                        // ---- vector fit (cosine on missing part) ----
                        dot      += need * w;
                        normNeed += need * need;
                        normNode += w * w;

                        // ---- emptiness magnitude ----
                        emptiness += need / WLmat[k][t];
                    }

                    // ---- normalize ----
                    double vectorFit = 0.0;
                    if (normNeed > 1e-12 && normNode > 1e-12)
                        vectorFit = dot / (sqrt(normNeed) * sqrt(normNode)); // [0,1]

                    emptiness /= M_weights; // [0,1]
                    overflow /= M_weights; // [0,1]

                    // ---- combine capacity info ----
                    double capacityGain =
                        0.8 * vectorFit +        // hợp hình
                        0.3 * emptiness;          // cụm đang đói
                    
                    double violationPenalty = exp(-4.0 * overflow); // vi phạm

                    // ---- distance heuristic ----
                    double distTerm = clusterSumDist[k][i] / DIST_SCALE;

                    double desir =
                        (1.0 / (1.0 + distTerm)) *
                        (1.0 + capacityGain) *
                        violationPenalty;

                    double tau = clusterSumPhero[k][i];
                    double weight = pow(tau, alpha) * pow(desir, beta);
                    weights[k] = weight;

                    if (weight > bestWeight)
                    {
                        bestWeight = weight;
                        chosenK = k;
                    }
                }

                // --- selection (exploitation / roulette) ---
                double q = uni01(rng);
                if (q >= Q0)
                {
                    double sumW = accumulate(weights.begin(), weights.end(), 0.0);
                    if (sumW > 0)
                    {
                        double pick = uni01(rng) * sumW;
                        double acc = 0.0;
                        for (int k = 0; k < K; ++k)
                        {
                            acc += weights[k];
                            if (pick <= acc)
                            {
                                chosenK = k;
                                break;
                            }
                        }
                    }
                }

                ants[a].assign[i] = chosenK;

                // cập nhật trọng số
                for (int t = 0; t < M_weights; ++t)
                clusterWeight[chosenK][t] += Wmat[i][t];

                // cập nhật chi phí
                for (int j = 0; j < N; ++j)
                    clusterSumDist[chosenK][j] += distmat[i][j];
                
                // update pheromone
                for (int j = 0; j < N; ++j)
                    clusterSumPhero[chosenK][j] += phi[i][j];

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

        vector<int> selected;  // index các ant được repair
        int MIN_DIFF = 1;
        for (int idx = 0; idx < m && selected.size() < repairTop; ++idx)
        {
            int ai = order[idx];
            bool diverse = true;

            for (int sj : selected)
            {
                if (hamming_distance(ants[ai].assign, ants[sj].assign) < MIN_DIFF)
                {
                    diverse = false;
                    break;
                }
            }

            if (diverse)
                selected.push_back(ai);
        }

        // nếu thiếu (trường hợp ants quá giống nhau)
        for (int idx = 0; idx < m && selected.size() < repairTop; ++idx)
        {
            int ai = order[idx];
            if (find(selected.begin(), selected.end(), ai) == selected.end())
                selected.push_back(ai);
        }

        // chạy local search
        for (int ai : selected)
        {
            local_search(ants[ai].assign, rng, 1000);
            large_search(ants[ai].assign, rng, 1000);
            ants[ai].cost = compute_cost(ants[ai].assign);
            ants[ai].feasible = is_feasible(ants[ai].assign);
        }

        // after local search, resort by feasibility then cost
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

                Q0 = Q_max;

                noImprove = 0;

                auto now = Clock::now();
                double elapsed = chrono::duration<double>(now - start).count();
                cerr << "[ITER " << iter << "] New best cost=" << format_cost_with_commas(best.cost, 0)
                     << " (feasible=" << (best.feasible ? "YES" : "NO") << ", time " << elapsed << "s)\n";
            }
        }
        if (!improvedThisIter)
        {
            ++noImprove;
            if (noImprove > STAGNATE_LIMIT)
            {
                Q0 -= STAGNATE_DROP;
                if (Q0 < Q_min)
                    Q0 = Q_min;
            }
        }

        // --- PHEROMONE UPDATE ---
        // Evaporation
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                phi[i][j] *= (1.0 - rho);

        // Best ant
        for (int i = 0; i < N; ++i)
        {
            for (int j = i + 1; j < N; ++j)
            {
                if (best.assign[i] == best.assign[j])
                {
                    phi[i][j] += T_max;
                    phi[j][i] += T_max;
                }
            }
        }

        // Find BEST LOCAL khác best global theo Hamming
        int bestLocal = -1;

        for (int r = 0; r < m; ++r)
        {
            int ai = order[r];

            // nếu giống best global thì bỏ
            if (hamming_distance(ants[ai].assign, best.assign) < MIN_DIFF)
                continue;

            bestLocal = ai;
            break;
        }

        // optionally, deposit Tmin for all ants
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (i != j)
                    phi[i][j] = max(phi[i][j], T_min);

        // clamp phi to avoid extremes
        const double PHI_MIN = 1e-6, PHI_MAX = 5;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (i != j)
                    phi[i][j] = max(PHI_MIN, min(PHI_MAX, phi[i][j]));

        // logging
        if (iter % 10 == 0)
        {
            auto bestThisIter = ants[bestLocal];
            int feasCount = 0;
            for (int a = 0; a < m; ++a)
            {
                if (ants[a].feasible)
                {
                    feasCount++;
                }
            }
            double elapsed = chrono::duration<double>(Clock::now() - start).count();
            cerr << "[ITER " << iter << "] bestGlobalCost=" << format_cost_with_commas(best.cost, 0)
                 << " (feasible=" << (best.feasible ? "YES" : "NO") << ")"
                 << " bestThisIter=" << format_cost_with_commas(bestThisIter.cost, 0)
                 << " feasibleAnts=" << feasCount
                 << " noImprove=" << noImprove
                 << " (elapsed " << elapsed << "s)\n";
            // for (int r = 0; r < repairTop; ++r) {
            //     cerr << "ant" << r << " cost = " << ants[selected[r]].cost << endl;
            // }
            // append snapshot to in-memory log
            LogRow r;
            r.iter = iter;
            r.time = elapsed;
            r.bestCost = best.cost;
            r.bestFeasible = best.feasible;
            r.bestThisIter = bestThisIter.cost;
            r.feasibleAnts = feasCount;
            r.noImprove = noImprove;
            log_rows.push_back(r);
        }

        // stagnation reset
        int noImproveReset = 400;
        if (noImprove >= noImproveReset)
        {
            cerr << "[RESET] no improvement for" << noImproveReset << "-> reset pheromones\n";
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < K; ++k)
                    phi[i][k] = T_min;
            noImprove = 0;
            Q0 = Q_max;
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
