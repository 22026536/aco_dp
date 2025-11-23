#pragma once
#include <bits/stdc++.h>
#include "../algorithm/Algorithm.h"
using namespace std;

struct ACOSolution {
    vector<int> assign;
    double cost = 1e300;
    bool feasible = false;
};

// Tunable parameters
extern double PENALTY_SCALE;
extern double OVERLOAD_PENALTY_FACTOR;
extern bool ALLOW_VIOLATIONS;
extern int MAX_ANTS;

// Global data
extern int N, K, M_weights;
extern vector<double> w1, w2;
extern vector<vector<double>> distmat;
extern vector<double> Wmin1, Wmax1, Wmin2, Wmax2;

// Functions
bool check_weights_validity(const Instance instance);
double compute_cost(const vector<int> &assign);
bool is_feasible(const std::vector<int> &assign);
void local_search(vector<int> &assign, mt19937_64 &rng, int maxMoves);
double cluster_violation_from_sums(const vector<vector<double>> &sumW, int k);
void repair_solution(vector<int> &assign, mt19937_64 &rng);
void improve_ant_solution(vector<int> &assign, mt19937_64 &rng, int repairPasses, int localPasses);
void SaveLogs(const ACOSolution &best);
ACOSolution ACO_tuned(const Instance &instance, int maxIter = 1000000, double timeLimitSeconds = 300.0, const string &instance_name = "");
