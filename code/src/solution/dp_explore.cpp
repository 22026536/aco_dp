#include "Solution.h"

#include "bits/stdc++.h"

bool Solution::ExploreDP() {

    const size_t pow2 = 1 << instance.nK;
    std::vector<std::vector<std::vector<Solution>>> dp(pow2);
    for (unsigned mask = 1; mask < pow2; ++mask) {
        dp[mask].resize(__builtin_popcount(mask));
    }
    std::vector<bool> inqueue(pow2);
    std::queue<unsigned> que;
    for (unsigned i = 0; i < instance.nK; i++) {
        unsigned m = (1u << i);
        que.emplace(m);
        inqueue[m] = true;
        dp[m][0].emplace_back(Solution(*this));
    }



    while (!que.empty()) {
        unsigned mask = que.front();
        que.pop();

        if (mask == (unsigned)(pow2 - 1)) break;

        unsigned len = __builtin_popcount(mask), pass = 0;

        for (unsigned i = 0; i < len; i++) {
            unsigned u = __builtin_ctz(mask ^ pass);
            pass |= 1 << u;

            for (unsigned v = 0, idx = 0; v < instance.nK; v++)
            if (mask >> v & 1) idx++;
            else {
                unsigned new_mask = mask | (1u << v);

                if (!inqueue[new_mask]) {
                    inqueue[new_mask] = true;
                    que.emplace(new_mask);
                }

                for (Solution &solution : dp[mask][i]) {

                    for (int pos = 0; pos < int(solution.sol[u].size()); ++pos) {

                        // Random shuffle to diversify the search
                        unsigned curSize = solution.sol[u].size();
                        if (pos + 1 < curSize) {
                            unsigned r = rndUns(pos, curSize - 1);
                            std::swap(solution.sol[u][pos], solution.sol[u][r]);
                        }
                        unsigned item = solution.sol[u][pos];

                        // Check feasibility of moving 'item' from cluster u to cluster v
                        if (!solution.FeasibleInsert(item, u, v)) {
                            continue;
                        }

                        double deltaObj = solution.EvaluateInsert(true, item, u, v);
                        if (deltaObj < 0) {
                            solution.obj += deltaObj;
                            if (parameters.LSe) {
                                solution.DataUpdateInsert(true, item, u, v);
                            }
                            solution.Insert(pos, u, solution.sol[v].size(), v);
                        }
                    }

                    if (dp[new_mask][idx].empty() or solution.obj < dp[new_mask][idx].back().obj) {
                        dp[new_mask][idx].emplace_back(solution);
                        for (int c = int(dp[new_mask][idx].size()) - 1; c > 0; c--) {
                            if (dp[new_mask][idx][c].obj < dp[new_mask][idx][c - 1].obj) {
                                std::swap(dp[new_mask][idx][c], dp[new_mask][idx][c - 1]);
                            } else {
                                break;
                            }
                        }
                        if (dp[new_mask][idx].size() > parameters.DPc) {
                            dp[new_mask][idx].pop_back();
                        }
                    }
                }
            }
        }
    }

    Solution bestSolution = *this;
    for (int last = 0; last < instance.nK; last++) {
        for (Solution &solution : dp[pow2 - 1][last]) {
            if (solution.obj < bestSolution.obj) {
                std::swap(bestSolution, solution);
            }
        }
    }

    if (bestSolution.obj < this->obj) {
        *this = bestSolution;
        return true;
    }

    return false;
}