/* 
 * File:   Source file for class Island
 * Author: A. Herrán
 */

#include "Island.h"

// Runtime debug controlled by parameters.DEBUG

//------------------------
// Constructor/destructor
//------------------------
Island::Island() {

    metaheuristics = new Metaheuristic *[parameters.ALGnm];
    if(parameters.ALGm == "grasp") for(unsigned m=0; m < parameters.ALGnm; m++) metaheuristics[m] =  new GRASP();
    else if(parameters.ALGm == "grasp-dp-opt") for(unsigned m=0; m < parameters.ALGnm; m++) metaheuristics[m] =  new GRASP();
    else {
        cout << "\n Wrong meta-heuristic selected.";
        cin.get();
        exit(1);
    }   
    
}

Island::~Island() { 
    
    for(unsigned m=0; m < parameters.ALGnm; m++) delete metaheuristics[m];
    delete[] metaheuristics;

}

//----------------------------------------------------------
// Performs one iteration of each one of the meta-heuristic
//----------------------------------------------------------
void Island::Iterate() {

    for(unsigned m=0; m < parameters.ALGnm; m++) metaheuristics[m]->Iterate();

}

//--------------------------------------------------------------------------------
// Adds the incoming front to the worst front among all the ALGnm meta-heuristics
//--------------------------------------------------------------------------------
void Island::Add(Solution &solution) {
       
 // Get the worst solutions among the ALGnm meta-heuristics
    unsigned iworst = 0;
    for(unsigned m=1; m<parameters.ALGnm; m++) if(metaheuristics[m]->solutione > metaheuristics[iworst]->solutione) iworst = m;

 // Replace the worst solution with the incoming one
    metaheuristics[iworst]->solutione = solution;
    metaheuristics[iworst]->k = 0;
    
}

//--------------------------------------------------------
// Gets the best solution among the ALGnm meta-heuristics
//--------------------------------------------------------
Solution Island::GetBest() {
    
    unsigned ibest = 0;
    for(unsigned m=1; m<parameters.ALGnm; m++) if(metaheuristics[m]->solutione < metaheuristics[ibest]->solutione) ibest = m;

    return metaheuristics[ibest]->solutione;
    
}
