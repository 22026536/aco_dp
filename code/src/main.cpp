/*
 * File:   main.cpp
 * Author: Quang
 * Description: ACO main program
 */

#include <bits/stdc++.h>
#include "aco/ACO.h"
#include "algorithm/Algorithm.h"
#include <chrono>
using Clock = std::chrono::steady_clock;
using namespace std;

// ---------------------------
// Global parameters and random engines
// ---------------------------
bool parallel_enabled = true;
vector<Tengine> vengine;

// Global instance and parameters
Instance instance;
Parameters parameters;
double termination_time = 300.0;
int maxIter = 1000000;
string instance_path;

// ---------------------------
// Function declarations
// ---------------------------
string LoadInput(int argc, const char *argv[], unsigned &seed);
void LoadInstance(const string &pathInstance);
void LogParameters();
void abstract();

// ---------------------------
// Main
// ---------------------------
int main(int argc, const char *argv[])
{

    unsigned seed = 0;
    instance_path = LoadInput(argc, argv, seed);
    LoadInstance(instance_path);
    // create desired folders and pass to parameters
    std::string aco_logdir = "results/logs/aco_logs";
    std::string mkcmd = "mkdir -p " + aco_logdir + "/evolution " + aco_logdir + "/solutions " + aco_logdir + "/objectives";
    system(mkcmd.c_str());
    parameters.LOGdir = aco_logdir;

    // Initialize random engine (like old code)
    random_device rd;
    for (unsigned p = 0; p < omp_get_max_threads(); p++)
    {
        vengine.emplace_back(rd());
        if (seed != 0)
            vengine[p].seed((p + 1) * seed);
        else
            vengine[p].seed(random_device{}());
    }

    // Log parameters and abstract
    abstract();
    LogParameters();

    // Run ACO
    std::string instance_file = instance_path.substr(instance_path.find_last_of("/\\") + 1);
    ACOSolution best = ACO_tuned(instance, maxIter, termination_time, instance_file);

    // Kiểm tra solution
    if (!best.feasible)
        cout << "Final solution is invalid.\n";
    else
        cout << "Final solution is valid.\n";

    cout << "Final cost = " << best.cost << "\n";
    return 0;
}

// ---------------------------
// Load input arguments
// ---------------------------
string LoadInput(int argc, const char *argv[], unsigned &seed)
{

    bool input;
    string pathInstance;
    for (int i = 1; i < argc; i += 2)
    {
        input = false;
        if (argv[i][0] == '-' && i + 1 < argc)
        {
            string key = argv[i];
            string value = argv[i + 1];

            if (key == "--instance")
            {
                input = true;
                pathInstance = value;
            }
            else if (key == "--termination_criteria")
            {
                input = true;
                parameters.ALGtc = value;
            }
            else if (key == "--seed")
            {
                input = true;
                seed = stoi(value);
            }
            else if (key == "--iter_value")
            {
                input = true;
                maxIter = stoi(value);
            }
            else if (key == "--termination_value")
            {
                input = true;
                termination_time = stod(value);
                parameters.ALGtv = stod(value);
            }
            else if (key == "--schema")
            {
                input = true;
                parameters.CONm = value;
            }
            else if (key == "--logs")
            {
                input = true;
                parameters.ALGlg = stoi(value);
            }
            else if (key == "--version")
            {
                input = true;
                parameters.GRASPv = value;
            }
            else if (key == "--alpha")
            {
                input = true;
                parameters.GRASPa = stod(value);
            }
            else if (key == "--m")
            {
                input = true;
                parameters.GRASPm = stoi(value);
            }
            else if (key == "--block")
            {
                input = true;
                parameters.GRASPb = stoi(value);
            }
            else if (key == "--delta")
            {
                input = true;
                parameters.GRASPd = stoi(value);
            }
            else if (key == "--move")
            {
                input = true;
                parameters.LSm = value;
            }
            else if (key == "--efficient")
            {
                input = true;
                parameters.LSe = (stoi(value) != 0);
            }
            else if (key == "--exploration")
            {
                input = true;
                parameters.LSs = value;
                if ((value != "best") && (value != "hybrid") && (value != "first"))
                {
                    cerr << "\nWrong exploration strategy.\n";
                    exit(1);
                }
            }
            else if (key == "--debug")
            {
                input = true;
                if ((value == "1") || (value == "true"))
                    parameters.DEBUG = true;
                else if ((value == "0") || (value == "false"))
                    parameters.DEBUG = false;
                else
                {
                    cerr << "\nWrong debug flag. Use 1/0 or true/false.\n";
                    exit(1);
                }
            }
        }

        if (!input)
        {
            cerr << "\nWrong input: " << argv[i] << "\n";
            exit(1);
        }
    }

    return pathInstance;
}

// ---------------------------
// Load instance from file
// ---------------------------
void LoadInstance(const string &pathInstance)
{

    ifstream file(pathInstance);
    if (!file)
    {
        cerr << "\nThe instance " << pathInstance << " cannot be opened.\n";
        exit(1);
    }

    file >> instance.type;

    if (instance.type == "p" || instance.type == "t")
    {
        file >> instance.nV >> instance.nK >> instance.nT;
        instance.W = matDbl(instance.nV, vecDbl(instance.nT, 0.0));
        instance.D = matDbl(instance.nV, vecDbl(instance.nV, 0.0));
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0));
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0));

        vector<vecDbl> C(instance.nV, vecDbl(2, 0.0));

        for (int i = 0; i < instance.nV; i++)
        {
            string ignore;
            file >> ignore;
            for (int t = 0; t < instance.nT; t++)
                file >> instance.W[i][t];

            if (instance.type == "p")
            {
                for (int j = 0; j < instance.nV; j++)
                {
                    file >> instance.D[i][j];
                    instance.D[i][j] /= 2.0;
                }
            }
            else
            {
                file >> C[i][0] >> C[i][1];
            }
        }

        if (instance.type == "t")
        {
            unsigned decimals = 6;
            double factor = pow(10, decimals);
            for (int i = 0; i < instance.nV; i++)
            {
                for (int j = i + 1; j < instance.nV; j++)
                {
                    instance.D[i][j] = round(sqrt(pow(C[i][0] - C[j][0], 2) + pow(C[i][1] - C[j][1], 2)) / 2.0 * factor) / factor;
                    instance.D[j][i] = instance.D[i][j];
                }
            }
        }

        // Read weight limits
        for (unsigned k = 0; k < instance.nK; k++)
        {
            for (unsigned t = 0; t < instance.nT; t++)
                file >> instance.WL[k][t];
            for (unsigned t = 0; t < instance.nT; t++)
                file >> instance.WU[k][t];
        }
    }
    else if (instance.type == "h")
    {
        double handover;
        file >> instance.nV >> instance.nK >> handover;
        instance.nT = 1;
        instance.W = matDbl(instance.nV, vecDbl(instance.nT, 0.0));
        instance.D = matDbl(instance.nV, vecDbl(instance.nV, 0.0));
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0));
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0));

        string ignore;
        for (unsigned k = 0; k < instance.nK; k++)
            file >> ignore >> ignore >> instance.WU[k][0];
        for (unsigned i = 0; i < instance.nV; i++)
            file >> ignore >> ignore >> instance.W[i][0];

        string line;
        getline(file, line);
        while (getline(file, line))
        {
            vector<string> tokens = Algorithm::Split(line, ' ');
            instance.D[stoi(tokens[1]) - 1][stoi(tokens[2]) - 1] = -stod(tokens[3]);
        }
    }
    else
    {
        cerr << "\nWrong instance file.\n";
        exit(1);
    }

    file.close();
}

void abstract()
{
    cout << "\n Structure: There are multiple islands, each containing one or more metaheuristic GRASP algorithms (by default, the code assigns one metaheuristic per island; this cannot be changed in the configuration file and must be modified directly in the code).\n";
    cout << "\n Operation: For each global iteration, one iteration is executed on every island. When an island performs an iteration, it means that all metaheuristics within that island also execute an iteration. When a metaheuristic performs an iteration, it refers to running the entire algorithm process once — including generating a new solution, validating it, and optimizing it.\n";
}

// ---------------------------
// Log parameterscerr
// ---------------------------
void LogParameters()
{
    cout << "\nAlgorithm parameters:";
    cout << "\n  - Search method: " << parameters.ALGm;
    cout << "\n  - Termination criteria: " << parameters.ALGtc << " = " << parameters.ALGtv;
    cout << "\n  - Number of islands: " << parameters.ALGni;
    cout << "\n  - Migration rate: " << parameters.ALGmr;
    cout << "\n  - Logs frequency: " << parameters.ALGlg;

    if (parameters.ALGm == "grasp")
    {
        cout << "\nMeta-heuristic parameters:";
        cout << "\n  - Schema: " << parameters.CONm;
        cout << "\n  - GRASP version: " << parameters.GRASPv;
        cout << "\n  - Alpha: " << parameters.GRASPa;
        cout << "\n  - Number of alphas (m): " << parameters.GRASPm;
        cout << "\n  - Block size: " << parameters.GRASPb;
        cout << "\n  - Delta iterations: " << parameters.GRASPd;
    }
    cout << endl;
}
