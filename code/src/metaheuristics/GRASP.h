/* 
 * File:   Header file for class GRASP
 * Author: A. Herrán
 */

#ifndef GRASP_H
#define GRASP_H

#include "algorithm/Metaheuristic.h"

class GRASP: public Metaheuristic {
    
    public:

     // Constructor/destructor
        GRASP();
        ~GRASP() override;
        
    private:

     // Private parameters
        double alpha;
        int    m, block, delta;
        vecUns n;
        vecDbl valpha, p, sum, q;

     // Private functions
        void Iterate() override;

     // Utils
        double myRound(double value) {
            unsigned decimals = 6;
            double factor = pow(10,decimals);
            return round(value*factor)/factor;
        }
        double rndDbl(double from, double upto) {
            Tengine &engine = vengine[omp_get_thread_num()];
            uniform_real_distribution<double> rndDist(from,upto);
            return rndDist(engine);
        }

};

#endif /* GRASP_H */
