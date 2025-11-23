#include "Solution.h"

void Solution::Exchange(unsigned i1, unsigned k1, unsigned i2, unsigned k2) {

    unsigned temp = sol[k1][i1];
    sol[k1][i1]   = sol[k2][i2];
    sol[k2][i2]   = temp;

}

bool Solution::ExploreExchange(bool objective) {

    unsigned i1, k1, i2, k2;
    
    double Dobj = ExploreExchangeMove(objective,i1,k1,i2,k2);
    if(Dobj<0.0) {
        obj += Dobj;
        if(parameters.LSe) DataUpdateExchange(objective,sol[k1][i1],k1,sol[k2][i2],k2);
        Exchange(i1,k1,i2,k2);
        return true;
    }

    return false;
}

double Solution::ExploreExchangeMove(bool objective, unsigned &i1best, unsigned &k1best, unsigned &i2best, unsigned &k2best) {

    double Dobjbest = 0.0;

    vector<vector<bool>> mat = vector<vector<bool>>(instance.nV,vector<bool>(instance.nV,true));
    for(unsigned v=0; v<instance.nV; v++) mat[v][v] = false;

    pairVK pair_ki;
    vector<pairVK> vecpair_ki;
    for(pair_ki.first=0; pair_ki.first < instance.nK; pair_ki.first++) {
        for(pair_ki.second=0; pair_ki.second < sol[pair_ki.first].size(); pair_ki.second++) {
            vecpair_ki.push_back(pair_ki);
        }
    }
    rndShuffle(vecpair_ki);

    bool improved = false;
    for(auto&& pair_k1i1: vecpair_ki) {
        unsigned k1 = pair_k1i1.first;
        unsigned i1 = pair_k1i1.second;
        unsigned v1 = sol[k1][i1];
        for(auto&& pair_k2i2: vecpair_ki) {
            unsigned k2 = pair_k2i2.first;
            unsigned i2 = pair_k2i2.second;
            unsigned v2 = sol[k2][i2];
            if(mat[v1][v2]&&(k1!=k2)) {
                mat[v1][v2] = false;
                mat[v2][v1] = false;
                double Dobj = 0.0;
                if(parameters.LSe) { 
                    if(objective) { if(FeasibleExchange(v1,k1,v2,k2)) Dobj = EvaluateExchange(objective,v1,k1,v2,k2); }
                    else                                              Dobj = EvaluateExchange(objective,v1,k1,v2,k2);
                }
                else {
                    Exchange(i1,k1,i2,k2);
                    if(objective) { if(FeasibleMove(k1,k2)) Dobj = EvaluateMove(objective); }
                    else                                    Dobj = EvaluateMove(objective);
                    Exchange(i2,k2,i1,k1);
                }
                if(Dobj<0) {
                    if(Dobj<Dobjbest || ((Dobj==Dobjbest)&&(rndUns(0,1)))) {
                        i1best   = i1;
                        k1best   = k1;
                        i2best   = i2;
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

    return Dobjbest;
}

bool Solution::FeasibleExchange(unsigned i1, unsigned k1, unsigned i2, unsigned k2) {

    for(unsigned t=0; t<instance.nT; t++) {
        double wkt = w[k1][t] - instance.W[i1][t] + instance.W[i2][t];
        if(wkt<instance.WL[k1][t]) return false;
        if(wkt>instance.WU[k1][t]) return false;
        wkt = w[k2][t] - instance.W[i2][t] + instance.W[i1][t];
        if(wkt<instance.WL[k2][t]) return false;
        if(wkt>instance.WU[k2][t]) return false;
    }

    return true;
}

double Solution::EvaluateExchange(bool objective, unsigned i1, unsigned k1, unsigned i2, unsigned k2) {

    double Dobj = 0.0;

    if(objective) Dobj =  delta[i1][k2] - delta[i1][k1] + delta[i2][k1] - delta[i2][k2] - 2*(instance.D[i1][i2] + instance.D[i2][i1]);
    else {
        double sigmaaux = 0.0;
        for(unsigned t=0; t<instance.nT; t++) {
            double wk1t = w[k1][t] - instance.W[i1][t] + instance.W[i2][t];
            double wk2t = w[k2][t] - instance.W[i2][t] + instance.W[i1][t];
            if(wk1t < instance.WL[k1][t]) sigmaaux += instance.WL[k1][t] - wk1t;
            if(wk1t > instance.WU[k1][t]) sigmaaux += wk1t - instance.WU[k1][t];
            if(wk2t < instance.WL[k2][t]) sigmaaux += instance.WL[k2][t] - wk2t;
            if(wk2t > instance.WU[k2][t]) sigmaaux += wk2t - instance.WU[k2][t];
        }
        Dobj = sigmaaux - sigma[k1] - sigma[k2];
    }

    return myRound(Dobj);
}

void Solution::DataUpdateExchange(bool objective, unsigned i1, unsigned k1, unsigned i2, unsigned k2) {

    if(objective) {
        for(unsigned j=0; j<instance.nV; j++) {
            double di1j = instance.D[i1][j] + instance.D[j][i1];
            double di2j = instance.D[i2][j] + instance.D[j][i2];
            delta[j][k1] = delta[j][k1] - di1j + di2j;
            delta[j][k2] = delta[j][k2] + di1j - di2j;
        }

        for(unsigned t=0; t<instance.nT; t++) {
            w[k1][t] = w[k1][t] - instance.W[i1][t] + instance.W[i2][t];
            w[k2][t] = w[k2][t] - instance.W[i2][t] + instance.W[i1][t];
        }
    }
    else {
        sigma[k1] = 0.0;
        sigma[k2] = 0.0;
        for(unsigned t=0; t<instance.nT; t++) {
            w[k1][t] = w[k1][t] - instance.W[i1][t] + instance.W[i2][t];
            w[k2][t] = w[k2][t] - instance.W[i2][t] + instance.W[i1][t];
            if(w[k1][t] < instance.WL[k1][t]) sigma[k1] += instance.WL[k1][t] - w[k1][t];
            if(w[k1][t] > instance.WU[k1][t]) sigma[k1] += w[k1][t] - instance.WU[k1][t];
            if(w[k2][t] < instance.WL[k2][t]) sigma[k2] += instance.WL[k2][t] - w[k2][t];
            if(w[k2][t] > instance.WU[k2][t]) sigma[k2] += w[k2][t] - instance.WU[k2][t];
        }
    }

}
