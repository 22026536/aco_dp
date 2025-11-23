#include "Solution.h"

#include "bits/stdc++.h"

// void Solution::Insert(unsigned i, unsigned k1, unsigned j, unsigned k2)  {

//     unsigned v = sol[k1][i];
//     sol[k2].insert(sol[k2].begin() + j, v);
//     sol[k1].erase(sol[k1].begin() + i);

// }

// This might be faster.
void Solution::Insert(unsigned i, unsigned k1, unsigned j, unsigned k2)  {

    sol[k2].emplace_back(sol[k1][i]);
    std::swap(sol[k2].back(), sol[k2][j]);
    std::swap(sol[k1][i], sol[k1].back());
    sol[k1].pop_back();

}

bool Solution::ExploreInsert(bool objective) {

    unsigned i, k1, k2;
    double   Dobj = ExploreInsertMove(objective,i,k1,k2);

    if(Dobj<0.0) {
        obj += Dobj;
        if(parameters.LSe) DataUpdateInsert(objective,sol[k1][i],k1,k2);
        Insert(i,k1,rndUns(0,sol[k2].size()),k2);
        return true;
    }

    return false;
}

double Solution::ExploreInsertMove(bool objective, unsigned &ibest, unsigned &k1best, unsigned &k2best) {

    double Dobjbest = 0.0;

    vecUns vec_i, vec_k1, vec_k2;
    for(unsigned k=0; k<instance.nK; k++) vec_k1.emplace_back(k); rndShuffle(vec_k1);
    for(unsigned k=0; k<instance.nK; k++) vec_k2.emplace_back(k); rndShuffle(vec_k2);

    bool improved = false;
    for(unsigned k1: vec_k1) {
        vec_i.clear();
        for(unsigned i=0; i<sol[k1].size(); i++) vec_i.emplace_back(i); rndShuffle(vec_i);
        for(unsigned i: vec_i) {
            for(unsigned k2: vec_k2) {
                if(k2 != k1) {
                    double Dobj = 0.0;
                    if(parameters.LSe) { 
                        if(objective) { if(FeasibleInsert(sol[k1][i],k1,k2)) Dobj = EvaluateInsert(objective,sol[k1][i],k1,k2); }
                        else                                                 Dobj = EvaluateInsert(objective,sol[k1][i],k1,k2);
                    }
                    else {
                        Insert(i,k1,sol[k2].size(),k2);
                        if(objective) { if(FeasibleMove(k1,k2)) Dobj = EvaluateMove(objective); }
                        else                                    Dobj = EvaluateMove(objective);
                        Insert(sol[k2].size()-1,k2,i,k1);
                    }
                    if(Dobj<0) {
                        if(Dobj<Dobjbest || ((Dobj==Dobjbest)&&(rndUns(0,1)))) {
                            ibest    = i;
                            k1best   = k1;
                            k2best   = k2;
                            Dobjbest = Dobj;
                            improved = true;
                        }
                    }
                }
                if(improved&&(parameters.LSs=="first")) break;
            }
            if(improved&&(parameters.LSs!="best")) break;
        }
        if(improved&&(parameters.LSs!="best")) break;
    }

    return Dobjbest;
}

bool Solution::FeasibleInsert(unsigned i, unsigned k1, unsigned k2) {

    for(unsigned t=0; t<instance.nT; t++) {
        if(w[k1][t] - instance.W[i][t] < instance.WL[k1][t]) return false;
        if(w[k2][t] + instance.W[i][t] > instance.WU[k2][t]) return false;
    }

    return true;
}

double Solution::EvaluateInsert(bool objective, unsigned i, unsigned k1, unsigned k2) {

    double Dobj = 0.0;

    if(objective) Dobj = delta[i][k2] - delta[i][k1];
    else {
        double sigmaaux = 0.0;
        for(unsigned t=0; t<instance.nT; t++) {
            double wk1t = w[k1][t] - instance.W[i][t];
            double wk2t = w[k2][t] + instance.W[i][t];
            if(wk1t < instance.WL[k1][t]) sigmaaux += instance.WL[k1][t] - wk1t;
            if(wk1t > instance.WU[k1][t]) sigmaaux += wk1t - instance.WU[k1][t];
            if(wk2t < instance.WL[k2][t]) sigmaaux += instance.WL[k2][t] - wk2t;
            if(wk2t > instance.WU[k2][t]) sigmaaux += wk2t - instance.WU[k2][t];
        }
        return Dobj = sigmaaux - sigma[k1] - sigma[k2];
    }

    return myRound(Dobj);
}

void Solution::DataUpdateInsert(bool objective, unsigned i, unsigned k1, unsigned k2) {

    if(objective) {
        for(unsigned j=0; j<instance.nV; j++) {
            double dij = instance.D[i][j] + instance.D[j][i];
            delta[j][k1] = delta[j][k1] - dij;
            delta[j][k2] = delta[j][k2] + dij;
        }

        for(unsigned t=0; t<instance.nT; t++) {
            w[k1][t] -= instance.W[i][t];
            w[k2][t] += instance.W[i][t];
        }
    }
    else {
        sigma[k1] = 0.0;
        sigma[k2] = 0.0;
        for(unsigned t=0; t<instance.nT; t++) {
            w[k1][t] -= instance.W[i][t];
            w[k2][t] += instance.W[i][t];
            if(w[k1][t] < instance.WL[k1][t]) sigma[k1] += instance.WL[k1][t] - w[k1][t];
            if(w[k1][t] > instance.WU[k1][t]) sigma[k1] += w[k1][t] - instance.WU[k1][t];
            if(w[k2][t] < instance.WL[k2][t]) sigma[k2] += instance.WL[k2][t] - w[k2][t];
            if(w[k2][t] > instance.WU[k2][t]) sigma[k2] += w[k2][t] - instance.WU[k2][t];
        }
    }

}
