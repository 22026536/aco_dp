#include "Solution.h"

// Runtime debug controlled by parameters.DEBUG

bool Solution::ExploreExtended(bool objective) {

 // Insert move
    unsigned i_ins, k1_ins, k2_ins;
    double   Dobj_ins = ExploreInsertMove(objective,i_ins,k1_ins,k2_ins);

 // Exchange move
    unsigned i1_exc, k1_exc, i2_exc, k2_exc;
    double   Dobj_exc = ExploreExchangeMove(objective,i1_exc,k1_exc,i2_exc,k2_exc);

 // Applies the best move if it improves the solution
    if((Dobj_ins<0.0)||(Dobj_exc<0.0)) {
        if(Dobj_ins<Dobj_exc) {
            obj += Dobj_ins;
            if(parameters.LSe) DataUpdateInsert(objective,sol[k1_ins][i_ins], k1_ins, k2_ins);
            Insert(i_ins,k1_ins,rndUns(0,sol[k2_ins].size()),k2_ins);
            if(parameters.DEBUG) cout << " Extended move - Insert selected: D_obj = " << Dobj_ins << endl;
            return true;
        }
        else {
            obj += Dobj_exc;
            if(parameters.LSe) DataUpdateExchange(objective,sol[k1_exc][i1_exc],k1_exc,sol[k2_exc][i2_exc],k2_exc);
            Exchange(i1_exc,k1_exc,i2_exc,k2_exc);
            if(parameters.DEBUG) cout << " Extended move - Exchange selected: D_obj = " << Dobj_exc << endl;
            return true;
        }
    }

    if(parameters.DEBUG) cout << " Extended move - No improving move found." << endl;
    return false;
}
