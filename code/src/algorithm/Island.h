/* 
 * File:   Header file for class Island
 * Author: A. Herrán
 */

#ifndef ISLAND_H
#define ISLAND_H

#include "metaheuristics/GRASP.h"

class Island {
    
    public:
        
     // Constructor/destructor   
        Island();
        virtual ~Island();
        
     // Public functions
        void     Iterate();
        void     Add(Solution &solution);
        Solution GetBest();
    
    private:
       
     // Private attributes
        Metaheuristic **metaheuristics;
        
};

#endif /* ISLAND_H */
