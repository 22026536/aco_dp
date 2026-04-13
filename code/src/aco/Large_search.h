#pragma once
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
using namespace std;

double cluster_violation_from_sums(const vector<vector<double>> &sumW, int k);
pair<vector<int>, vector<int>> compute_over_under_from_sums(const vector<vector<double>> &sumW);
void large_search(vector<int> &clusterOfNode, mt19937_64 &rng, int MAX_MOVES);
