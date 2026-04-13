#include "ACO.h"
#include "Tabu_search.h"

void tabu_search(ACOSolution &sol, int maxIter, int tabuTenure, mt19937_64 &rng)
{
    auto &assign = sol.assign;
    auto &members = sol.members;
    auto &sumW = sol.clusterWeight;
    auto &sumDist = sol.clusterSumDist;

    vector<int> bestAssign = assign;
    double currentCost = sol.cost;
    double bestCost = currentCost;

    vector<vector<int>> tabu(N, vector<int>(K, 0));

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

                double delta = PENALTY_SCALE * deltaViol + deltaDist;

                bool isTabu = tabu[u][to] > iter;

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

        int u = bestU;
        int from = bestFrom;
        int to = bestTo;

        // ===== update members =====
        auto it = find(members[from].begin(), members[from].end(), u);
        if (it != members[from].end())
            members[from].erase(it);
        members[to].push_back(u);

        // ===== update weight =====
        for (int t = 0; t < M_weights; ++t)
        {
            sumW[from][t] -= Wmat[u][t];
            sumW[to][t] += Wmat[u][t];
        }

        // ===== update assign =====
        assign[u] = to;

        // ===== update sumDist (incremental) =====
        for (int v = 0; v < N; ++v)
        {
            if (v == u) continue;

            sumDist[v][from] -= distmat[v][u];
            sumDist[v][to] += distmat[v][u];
        }

        // update riêng u
        for (int k = 0; k < K; ++k)
        {
            double s = 0.0;
            for (int m : members[k])
                if (m != u)
                    s += distmat[u][m];
            sumDist[u][k] = s;
        }

        // ===== tabu =====
        tabu[u][from] = iter + tabuTenure + (rng() % 5);

        currentCost += bestDelta;

        if (currentCost < bestCost)
        {
            bestCost = currentCost;
            bestAssign = assign;
        }
    }

    // ===== restore best =====
    assign = bestAssign;

    // 🔥 QUAN TRỌNG: rebuild lại sol cho bestAssign
    // (vì bạn đã mutate trong quá trình search)
    sol.cost = compute_cost(assign);
    sol.feasible = is_feasible(assign);

    // nếu muốn tối ưu nữa: bạn có thể giữ snapshot tốt nhất luôn
}

void perturb(ACOSolution &sol, mt19937_64 &rng, int strength)
{
    auto &assign = sol.assign;
    auto &members = sol.members;
    auto &sumW = sol.clusterWeight;
    auto &sumDist = sol.clusterSumDist;

    for (int it = 0; it < strength; ++it)
    {
        int u = rng() % N;
        int from = assign[u];
        int to = rng() % K;

        if (from == to) continue;

        // ===== update members =====
        auto itf = find(members[from].begin(), members[from].end(), u);
        if (itf != members[from].end())
            members[from].erase(itf);
        members[to].push_back(u);

        // ===== update weight =====
        for (int t = 0; t < M_weights; ++t)
        {
            sumW[from][t] -= Wmat[u][t];
            sumW[to][t] += Wmat[u][t];
        }

        // ===== update assign =====
        assign[u] = to;

        // ===== update sumDist =====
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
    }

    // cập nhật lại cost
    sol.cost = compute_cost(sol.assign);
    sol.feasible = is_feasible(sol.assign);
}

void iterated_tabu_search(ACOSolution &sol, mt19937_64 &rng)
{
    ACOSolution bestSol = sol;
    double bestCost = sol.cost;

    for (int round = 0; round < 20; ++round)
    {
        tabu_search(sol, 100, 10, rng);

        double cost = sol.cost;  // đã update trong tabu

        if (cost < bestCost)
        {
            bestCost = cost;
            bestSol = sol;
        }

        perturb(sol, rng, N / 20);
    }

    sol = bestSol;
}
