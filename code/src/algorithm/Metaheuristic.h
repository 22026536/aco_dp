/* 
 * File:   Header file for class Metaheuristic
 * Author: A. Herrán
 */

#ifndef METAHEURISTIC_H
#define METAHEURISTIC_H

#include "solution/Solution.h"

class Metaheuristic {
    
    public:

     // Constructor/destructor   
        Metaheuristic();
        virtual ~Metaheuristic();

     // Public attributes
        int      k = 0;
        Solution solutione;

     // Public functions
        virtual void Iterate() = 0;

    protected:
        
     // Protected attributes
        Solution solutiona;
        
};

#endif /* METAHEURISTIC_H */
