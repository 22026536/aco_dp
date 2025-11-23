/* 
 * File:   includes.h
 * Author: A. Herrán
 */

#ifndef INCLUDES_H
#define INCLUDES_H

 // Libraries
    #include <iostream>     // cin()/cout()
    #include <filesystem>   // create_directory()
    #include <fstream>      // file streams
    #include <iomanip>      // io manipulators
    #include <cstdio>       // printf(), fprintf(), fopen()
    #include <cmath>        // exp(), pow(), sqrt()
    #include <vector>       // vector class
    #include <limits>       // std::numeric_limits<double>::infinity()
    #include <algorithm>    // sort(), random_shuffle()
    #include <omp.h>        // parallel execution
    #include <random>       // engines for random number generation
    #include <chrono>       // timer
    #include <array>

 // Type definitions
    using namespace std;
    using namespace std::chrono;
    
    //using Tengine = default_random_engine;
    //using Tengine = minstd_rand0;
    //using Tengine = minstd_rand;
    using Tengine = mt19937;
    //using Tengine = mt19937_64;
    //using Tengine = ranlux24_base;
    //using Tengine = ranlux48_base;
    //using Tengine = ranlux24;
    //using Tengine = ranlux48;
    //using Tengine = knuth_b;
    
    using clk     = std::chrono::high_resolution_clock;
    
    using vecUns  = vector<unsigned>;
    using matUns  = vector<vecUns>;

    using vecDbl  = vector<double>;
    using matDbl  = vector<vecDbl>;

    using pairVK  = pair<unsigned,unsigned>;
    using pairECL = pair<double,pairVK>;
    using vecECL  = vector<pairECL>;

 // Problem parameters (global variables)
    extern bool            parallel_enabled;
    extern vector<Tengine> vengine;

    struct Instance {
        string   type;  // Instance type: pollster (p), tsplib (t), hmp (h)
        unsigned nV;    // Number of vertices
        unsigned nK;    // Number of clusters
        unsigned nT;    // Number of attributes
        matDbl   D;     // Distance matrix
        matDbl   W;     // Weight of attributes for each node
        matDbl   WL;    // Lower bound for the weight of attributes in each cluster
        matDbl   WU;    // Upper bound for the weight of attributes in each cluster
    };

    struct Parameters {
      string LOGdir = "results/logs/aco_logs";
     // Algorithm
        string   ALGm   = "grasp";    // Search: grasp, ils, vns, sa, ts
        int      ALGni  = 1;          // Number of Islands
        int      ALGnm  = 1;          // Number of Meta-heuristics on each Island
        string   ALGms  = "none";     // Migration Strategy: none, ring, fully, master
        int      ALGmr  = 1;          // Best solutions migrates at every N·ALGmr iterations
        string   ALGtc  = "iter";     // Termination criteria: iter, tcpu
        double   ALGtv  = 1000;       // Termination value: number of iterations (ALGtc=iter), time in seconds (ALGtc=tcpu)
        int      ALGlg  = 1;          // Logs: Cost metric in iRace (0), Area metric in iRace (-1), a value>ALGtv saves console logs in a file
        bool     DEBUG  = false;     // Runtime debug flag (use --debug 1 to enable)
     // Constructive
        string   CONm   = "2P-R-GRASP"; // Construction: 2P-GRASP, 2P-R-GRASP, Rnd-GRASP, 2P-HGRASP-DP, 2P-GRASP-DP
     // Local Search
        string   LSm    = "ext_i";    // Local Search move: insert (ins), exchange (exc), extended (ext), ins_i, exc_i, ext_i
        bool     LSe    = true;       // Local Search efficient: true, false
        string   LSs    = "first";    // Local Search strategy: first, hybrid, best
     // GRASP
        string   GRASPv = "rnd-grd";  // Greedy version: grd-rnd, rnd-grd
        double   GRASPa = 0.50;       // Alpha coefficient controlling the greedy-random ratio: purely random (1), purely greedy (0)
        int      GRASPm = 4;          // Alpha division for Reactive GRASP
        int      GRASPb = 38;         // Block of iterations to update probabilities for Reactive GRASP
        int      GRASPd = 1;          // Delta exponent for Reactive GRASP
     // TS
        int      TSn    = 10;         // Maximum number of elements in the Tabu list
     // DP
        int      DPc    = 1;          // Number of best candidates to consider in DP layers
    };

    extern Parameters parameters;
    extern Instance   instance;
    
#endif /* INCLUDES_H */
