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

void tabu_search(
    ACOSolution &sol,
    int maxIter,
    int tabuTenure,
    mt19937_64 &rng);

void iterated_tabu_search(
    ACOSolution &sol,
    mt19937_64 &rng);
