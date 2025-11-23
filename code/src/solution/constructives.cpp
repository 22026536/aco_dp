#include "Solution.h"

//******************************************************************************
// ConstructiveGRD procedures
//******************************************************************************

// Runtime debug controlled by parameters.DEBUG

void Solution::ConstructiveRND() {

 // Generates a random ordering of vertex in CL
    vecUns CL;
    for(unsigned v=0; v<instance.nV; v++) CL.emplace_back(v);
    rndShuffle(CL);

 // Fills each cluster with V/K nodes from CL
    unsigned minSize = floor(instance.nV/instance.nK);
    for(unsigned k=0; k<instance.nK; k++) {
        for(unsigned i=0; i<minSize; i++) {
            sol[k].emplace_back(CL.back());
            CL.pop_back();
        }
    }

 // Adds the remaining nodes to the clusters
    while(CL.size()>0) {
        unsigned k = rndUns(0,instance.nK-1);
        sol[k].emplace_back(CL.back());
        CL.pop_back();
    }

    if(parameters.DEBUG) {
    cout << " Random construction: " << endl;
    for(unsigned k=0; k<instance.nK; k++) {
        cout << "  Cluster " << k << ": ";
        for(unsigned v: sol[k]) cout << v << " ";
        cout << endl;
    }
    }

}

void Solution::ConstructiveGRD(double alpha){

 // Generates a random ordering of nodes in CL
    vecUns CL;
    for(unsigned v=0; v<instance.nV; v++) CL.emplace_back(v);
    rndShuffle(CL);

 // Adds one node to each cluster at random
    for(unsigned k=0; k<instance.nK; k++) {
        sol[k].emplace_back(CL.back());
        CL.pop_back();
    }

 // Generates the Extended Candidate List of pairs [g,pair_ik]
    vecECL  ECL;
    pairECL candidate;
    for(unsigned v: CL) {
        candidate.first        = 0.0;
        candidate.second.first = v;
        for(unsigned k=0; k<instance.nK; k++) {
            candidate.second.second = k;
            ECL.emplace_back(candidate);
        }
    }

 // Adds nodes to clusters according to the Greedy function value
    int gik;
    while(!ECL.empty()) {
        rndShuffle(ECL);
        if     (parameters.GRASPv=="grd-rnd") gik = getGrdRnd(ECL, alpha);
        else if(parameters.GRASPv=="rnd-grd") gik = getRndGrd(ECL, alpha);
        else {
            cout << "\n Wrong greedy version.";
            cin.get();
            exit(1);
        }
        unsigned v = ECL[gik].second.first;
        unsigned k = ECL[gik].second.second;
        sol[k].emplace_back(v);
        for(gik=ECL.size()-1; gik > -1 ; gik--) if(ECL[gik].second.first == v) ECL.erase(ECL.begin() + gik);
    }

    if(parameters.DEBUG) {
    cout << " GRASP construction (alpha=" << alpha << "): " << endl;
    for(unsigned k=0; k<instance.nK; k++) {
        cout << "  Cluster " << k << ": ";
        for(unsigned v: sol[k]) cout << v << " ";
        cout << endl;
    }
    }
}

unsigned Solution::getGrdRnd(vecECL &ECL, double alpha) {

    double gmin = std::numeric_limits<double>::max();
    double gmax = std::numeric_limits<double>::min();

    for(auto&& pairECL: ECL) {
        pairECL.first = EvaluateGreedy(pairECL.second);
        if(pairECL.first<gmin) gmin = pairECL.first;
        if(pairECL.first>gmax) gmax = pairECL.first;
    }

    vecUns RCL;
    double th = gmin + alpha*(gmax-gmin);
    for(unsigned gik=0; gik<ECL.size(); gik++) if(ECL[gik].first <= th) RCL.emplace_back(gik);

    return RCL[rndUns(0, RCL.size()-1)];
}

unsigned Solution::getRndGrd(vecECL &ECL, double alpha){

    unsigned s    = max((1.0-alpha)*ECL.size(),1.0);
    double   gmin = std::numeric_limits<double>::max();

    unsigned gikbest;
    for(unsigned gik=0; gik<s; gik++) {
        ECL[gik].first = EvaluateGreedy(ECL[gik].second);
        if(ECL[gik].first < gmin || (ECL[gik].first == gmin && rndUns(0,1))) {
            gmin   = ECL[gik].first;
            gikbest = gik;
        }
    }

    return gikbest;
}

double Solution::EvaluateGreedy(pairVK pair_ik) {

    double g = 0;
    for(unsigned j: sol[pair_ik.second]) g += instance.D[pair_ik.first][j] + instance.D[j][pair_ik.first];

    return g;
}
