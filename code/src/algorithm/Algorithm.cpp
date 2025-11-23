/* 
 * File:   Source file for class Algorithm
 * Author: A. Herrán
 */

#include "Algorithm.h"

//------------------------
// Constructor/destructor
//------------------------
Algorithm::Algorithm(const string& pathInstance, unsigned ne) {

    islands = new Island *[parameters.ALGni];
    for(unsigned i=0; i < parameters.ALGni; i++) islands[i] = new Island();

    if(parameters.ALGlg > 0) {
        vector<string> tokens = Split(pathInstance + '/', '/');
        tokens = Split(tokens.back(), '.');
        string instance = tokens[0];

        // Route logs into dp_logs when using a construction schema that ends with "-DP",
        // otherwise use pred_logs. Keep the same subfolders (evolution, solutions, objectives).
        string baseDir = "results/logs/pred_logs/";

        if (parameters.CONm == "2P-GRASP-DP") {
            baseDir = "results/logs/dp_logs/";
        }
        else if (parameters.CONm == "2P-HGRASP-DP") {
            baseDir = "results/logs/hdp_logs/";
        }

        fileEvol.open(baseDir + "evolution/" + instance + "_" + to_string(ne) + "_" + parameters.CONm + ".txt");
        if (!fileEvol) {
            cout << "\n The fileEvol file can not be opened.";
            exit(1);
        }

        fileSolu.open(baseDir + "solutions/" + instance + "_" + to_string(ne) + ".txt");
        if (!fileSolu) {
            cout << "\n The fileSolu file can not be opened.";
            exit(1);
        }

        fileCost.open(baseDir + "objectives/" + instance + "_" + to_string(ne) + ".txt");
        if (!fileEvol) {
            cout << "\n The fileCost file can not be opened.";
            exit(1);
        }

    }
    
}

Algorithm::~Algorithm() {
    
    for(unsigned i=0; i < parameters.ALGni; i++) delete islands[i];
    delete[] islands;

    if(parameters.ALGlg > 0) {
        fileEvol.close();
        fileCost.close();
        fileSolu.close();
    }

}

//---------------------------------------
// Performs one iteration of each island
//---------------------------------------
void Algorithm::Iterate() {

    #pragma omp parallel for if(parallel_enabled) //private(instance,parameters)
    for(unsigned i=0; i<parameters.ALGni; i++) islands[i]->Iterate();

}

//-----------------------------
// Communication among islands
//-----------------------------
void Algorithm::Migrate() {

    if(parameters.ALGni > 1) {
        if     (parameters.ALGms == "none"  ) ;
        else if(parameters.ALGms == "ring"  ) Ring();
        else if(parameters.ALGms == "fully" ) Fully();
        else if(parameters.ALGms == "master") Master();
        else {
            cout <<"\n Wrong migration method.";
            cin.get();
            exit(1);
        }
        if(parameters.ALGms != "none"  ) cout << "*";
    }
   
}

void Algorithm::Ring() {
   
    Solution vsolution[parameters.ALGni];
    for(unsigned i=0; i < parameters.ALGni; i++) vsolution[i] = GetBest(i);

    islands[0]->Add(vsolution[parameters.ALGni - 1]);
    islands[0]->Add(vsolution[1]);
    for(unsigned i=1; i < parameters.ALGni - 1; i++) {
        islands[i]->Add(vsolution[i-1]);
        islands[i]->Add(vsolution[i+1]);
    }
    islands[parameters.ALGni - 1]->Add(vsolution[parameters.ALGni - 2]);
    islands[parameters.ALGni - 1]->Add(vsolution[0]);
                    
}

void Algorithm::Fully() {

    Solution vsolution[parameters.ALGni];
    for(unsigned i=0; i < parameters.ALGni; i++) vsolution[i] = GetBest(i);
    
    for(unsigned i1=0; i1 < parameters.ALGni; i1++)
        for(unsigned i2=0; i2 < parameters.ALGni; i2++)
            if(i2!=i1) islands[i2]->Add(vsolution[i1]);
                        
}

void Algorithm::Master() {

    Solution vsolution[parameters.ALGni];
    for(unsigned i=0; i < parameters.ALGni; i++) vsolution[i] = GetBest(i);

    unsigned ibest = 0;
    for(unsigned i=1; i < parameters.ALGni; i++) if(vsolution[i] < vsolution[ibest]) ibest = i;
    
    for(unsigned i=0; i < parameters.ALGni; i++) if(i != ibest) islands[i]->Add(vsolution[ibest]);
    if(parameters.ALGnm > 1) for(unsigned i=0; i < parameters.ALGni; i++) if(i != ibest) islands[ibest]->Add(vsolution[i]);
    
}

//-------------------------------------
// Gets the best solution ini island i
//-------------------------------------
Solution Algorithm::GetBest(unsigned i) {

    return islands[i]->GetBest();

}

//---------------------------------------------
// Gets the best solution among the ALGni islands
//---------------------------------------------
Solution Algorithm::GetBest() {

    Solution vsolution[parameters.ALGni];
    for(unsigned i=0; i < parameters.ALGni; i++) vsolution[i] = GetBest(i);
    
    unsigned ibest = 0;
    for(unsigned i=1; i < parameters.ALGni; i++)
        if(vsolution[i]<vsolution[ibest]) ibest = i;
    
    return vsolution[ibest];
    
}

//-------------------------------------------
// Prints the algorithm evolution to console
//-------------------------------------------
void Algorithm::PrintLogs(double tcpu, unsigned it) {

 // Records the algorithm evolution
    vtcpu.emplace_back(tcpu);
    vecDbl vcost = vecDbl(parameters.ALGni, 0.0);
    if(instance.type=="h") for(unsigned i=0; i < parameters.ALGni; i++) vcost[i] = GetBest(i).EvaluateHMP();
    else                   for(unsigned i=0; i < parameters.ALGni; i++) vcost[i] = GetBest(i).obj;
    mcost.emplace_back(vcost);

 // Prints the algorithm evolution to console
    if(it % parameters.ALGlg == 0) {
        cout << " Iteration" << setw(12) << it << "     Cost";
        for(unsigned i=0; i < parameters.ALGni; i++) cout << setw(12) << fixed << setprecision(2) << vcost[i];
        cout << endl << setprecision(6);

        if(parameters.DEBUG) {
        cout << "---------------------------------------------------------" << endl;
        cout << " Time" << setw(15) << fixed << tcpu << " seconds" << endl;

        cout << " Best solution per island:" << endl;
        for(unsigned i=0; i < parameters.ALGni; i++) {
            Solution solu = GetBest(i);
            cout << "  Island " << i << " - Cost: " << fixed << setprecision(2);
            if(instance.type=="h") cout << solu.EvaluateHMP() << " - Solution: ";
            else                   cout << solu.obj           << " - Solution: ";
            cout << endl;
        }

        cout << "---------------------------------------------------------" << endl;

        }
    }

}

//--------------------------------------------------------------------
// Saves the algorithm evolution and best cost and solution to a file
//--------------------------------------------------------------------
void Algorithm::SaveLogs() {

 // Saves the algorithm evolution to a file
    for(unsigned it=0; it<vtcpu.size(); it++) {
        fileEvol << setw(10) << fixed << setprecision(4) << vtcpu[it];
        for(unsigned i=0; i < parameters.ALGni; i++) fileEvol << setw(12) << fixed << setprecision(2) << mcost[it][i];
        fileEvol << endl;
    }

 // Saves the best cost to a file
    if(instance.type=="h") fileCost << fixed << setprecision(2) << GetBest().EvaluateHMP();
    else                   fileCost << fixed << setprecision(2) << GetBest().obj;

 // Saves the best solution to a file
    Solution solution = GetBest();
    for(unsigned k=0; k < instance.nK; k++) {
        sort(solution.sol[k].begin(), solution.sol[k].end());
        for(unsigned v: solution.sol[k]) fileSolu << v + 1 << " ";
        fileSolu << endl;
    }

}

//--------------------------------
// Splits a string by a separator 
//--------------------------------
vector<string> Algorithm::Split(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;
    while(getline(ss,token,delimiter)) tokens.push_back(token);
    return tokens;
}
