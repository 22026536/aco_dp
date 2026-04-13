#include "ACO.h"
#include "Tabu_search.h"

// ===== TABU SEARCH =====
void tabu_search(vector<int> &assign, int maxIter, int tabuTenure, mt19937_64 &rng)
{
    // ===== INIT =====
    vector<int> bestAssign = assign;

    vector<vector<int>> tabu(N, vector<int>(K, 0));

    vector<vector<int>> members(K);
    vector<vector<double>> sumW(K, vector<double>(M_weights, 0.0));
    vector<vector<double>> sumDist(N, vector<double>(K, 0.0));

    // build initial
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        members[c].push_back(i);

        for (int t = 0; t < M_weights; ++t)
            sumW[c][t] += Wmat[i][t];
    }

    for (int k = 0; k < K; ++k)
        for (int j : members[k])
            for (int i = 0; i < N; ++i)
                sumDist[i][k] += distmat[i][j];

    // ===== init cost =====
    double currentCost = compute_cost(assign);
    double bestCost = currentCost;

    // ===== LOOP =====
    for (int iter = 1; iter <= maxIter; ++iter)
    {
        double bestDelta = 1e300;
        int bestU = -1, bestFrom = -1, bestTo = -1;

        for (int u = 0; u < N; ++u)
        {
            int from = assign[u];

            for (int to = 0; to < K; ++to)
            {
                if (to == from) continue;

                // ===== VIOLATION DELTA =====
                double before = 0.0, after = 0.0;

                for (int t = 0; t < M_weights; ++t)
                {
                    double s_from = sumW[from][t];
                    double s_to = sumW[to][t];

                    if (s_from < WLmat[from][t]) before += WLmat[from][t] - s_from;
                    if (s_from > WUmat[from][t]) before += s_from - WUmat[from][t];
                    if (s_to < WLmat[to][t]) before += WLmat[to][t] - s_to;
                    if (s_to > WUmat[to][t]) before += s_to - WUmat[to][t];

                    double ns_from = s_from - Wmat[u][t];
                    double ns_to = s_to + Wmat[u][t];

                    if (ns_from < WLmat[from][t]) after += WLmat[from][t] - ns_from;
                    if (ns_from > WUmat[from][t]) after += ns_from - WUmat[from][t];
                    if (ns_to < WLmat[to][t]) after += WLmat[to][t] - ns_to;
                    if (ns_to > WUmat[to][t]) after += ns_to - WUmat[to][t];
                }

                double deltaViol = after - before;

                // ===== DIST DELTA =====
                double deltaDist = sumDist[u][to] - sumDist[u][from];

                // ===== TOTAL DELTA COST =====
                double delta = PENALTY_SCALE * deltaViol + deltaDist;

                bool isTabu = tabu[u][to] > iter;

                // aspiration
                if (isTabu && currentCost + delta >= bestCost)
                    continue;

                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestU = u;
                    bestFrom = from;
                    bestTo = to;
                }
            }
        }

        if (bestU == -1) break;

        // ===== APPLY MOVE =====
        int u = bestU;
        int from = bestFrom;
        int to = bestTo;

        // update members
        auto it = find(members[from].begin(), members[from].end(), u);
        if (it != members[from].end())
            members[from].erase(it);
        members[to].push_back(u);

        // update weight
        for (int t = 0; t < M_weights; ++t)
        {
            sumW[from][t] -= Wmat[u][t];
            sumW[to][t] += Wmat[u][t];
        }

        // update assign
        assign[u] = to;

        // update sumDist
        for (int v = 0; v < N; ++v)
        {
            if (v == u) continue;

            sumDist[v][from] -= distmat[v][u];
            sumDist[v][to] += distmat[v][u];
        }

        for (int k = 0; k < K; ++k)
        {
            double s = 0.0;
            for (int m : members[k])
                if (m != u)
                    s += distmat[u][m];
            sumDist[u][k] = s;
        }

        // update tabu
        tabu[u][from] = iter + tabuTenure + (rng() % 5);

        // update cost (incremental)
        currentCost += bestDelta;

        // update best
        if (currentCost < bestCost)
        {
            bestCost = currentCost;
            bestAssign = assign;
        }
    }

    assign = bestAssign;
}

void perturb(vector<int> &assign, mt19937_64 &rng, int strength)
{
    for (int i = 0; i < strength; ++i)
    {
        int u = rng() % N;
        assign[u] = rng() % K;
    }
}

void iterated_tabu_search(vector<int> &assign, mt19937_64 &rng)
{
    vector<int> best = assign;
    double bestCost = compute_cost(assign);

    for (int round = 0; round < 20; ++round)
    {
        tabu_search(assign, 100, 10, rng);

        double cost = compute_cost(assign);

        if (cost < bestCost)
        {
            bestCost = cost;
            best = assign;
        }

        perturb(assign, rng, N / 20);
    }

    assign = best;
}
