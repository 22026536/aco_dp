#pragma once
#include <bits/stdc++.h>
#include "../algorithm/Algorithm.h"
using namespace std;

struct ACOSolution {
    double cost = 1e300;
    bool feasible = false;
    vector<int> assign;
    vector<vector<int>> members;
    vector<vector<double>> clusterWeight;
    vector<vector<double>> clusterSumDist;
};

// ================== GLOBALS ==================
extern vector<int> log_iter;
extern vector<double> log_time;

struct LogRow {
    int iter;
    double time;
    double bestCost;
    bool bestFeasible;
    double bestThisIter;
    int feasibleAnts;
    int noImprove;
};
extern vector<LogRow> log_rows;

extern Parameters parameters;

extern string LOG_EVOL_FILENAME;
extern string LOG_COST_FILENAME;
extern string LOG_SOLU_FILENAME;

// Global data
extern int N;
extern int K;
extern int M_weights;

extern vector<vector<double>> Wmat;
extern vector<vector<double>> WLmat;
extern vector<vector<double>> WUmat;
extern vector<vector<double>> distmat;

// Tunable parameters
extern double PENALTY_SCALE;
extern double VALID_EPS;

// Functions
static inline std::string format_cost_fixed(double v, int decimals = 0);

static inline std::string format_cost_with_commas(double v, int decimals = 0);

bool check_weights_validity(const Instance instance);

double compute_cost(const vector<int> &assign);

bool is_feasible(const std::vector<int> &assign);

int hamming_distance(const vector<int>& a, const vector<int>& b);

void SaveLogs(const ACOSolution &best);

ACOSolution ACO_tuned(const Instance &instance, int maxIter = 1000000, double timeLimitSeconds = 300.0, const string &instance_name = "");
