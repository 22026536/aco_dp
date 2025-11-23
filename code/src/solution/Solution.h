/* 
 * File:   Header file for class Solution
 * Author: A. Herrán
 */

#ifndef SOLUTION_H
#define	SOLUTION_H

#include "../includes.h"

class Solution {
        
    public:
                
     // Constructor/destructor
        Solution();
        virtual ~Solution();

     // Operators redefinition
        bool     operator<(const Solution &solution) const { return obj<solution.obj; }
        bool     operator>(const Solution &solution) const { return obj>solution.obj; }

     // Public attributes
        matUns   sol;
        double   obj;

     // Public functions
        void     Construct(double alpha);
        bool     DPExplore();
        void     Evaluate(bool objective);
        bool     Validate();
        double   EvaluateHMP();
        bool     Explore(bool objective);
        void     DataInit();

    protected:

     // Private attributes
        matDbl   delta;
        vecDbl   sigma;
        matDbl   w;

     // Construction methods
        void     ConstructiveRND();
        void     ConstructiveGRD(double alpha);
        unsigned getGrdRnd(vecECL &ECL, double alpha);
        unsigned getRndGrd(vecECL &ECL, double alpha);
        double   EvaluateGreedy(pairVK pair_ik);

     // Evaluation of a move
        double   EvaluateMove(bool objective);
        bool     FeasibleMove(unsigned k1, unsigned k2);

     // Insert move
        void     Insert           (unsigned i, unsigned k1, unsigned j, unsigned k2);
        bool     ExploreInsert    (bool objective);
        double   ExploreInsertMove(bool objective, unsigned &ibest, unsigned &k1best, unsigned &k2best);
        double   EvaluateInsert   (bool objective, unsigned i, unsigned k1, unsigned k2);
        void     DataUpdateInsert (bool objective, unsigned i, unsigned k1, unsigned k2);
        bool     FeasibleInsert   (unsigned i, unsigned k1, unsigned k2);
        
     // Exchange move
        void     Exchange           (unsigned i1, unsigned k1, unsigned i2, unsigned k2);
        bool     ExploreExchange    (bool objective);
        double   ExploreExchangeMove(bool objective, unsigned &i1best, unsigned &k1best, unsigned &i2best, unsigned &k2best);
        double   EvaluateExchange   (bool objective, unsigned i1, unsigned k1, unsigned i2, unsigned k2);
        void     DataUpdateExchange (bool objective, unsigned i1, unsigned k1, unsigned i2, unsigned k2);
        bool     FeasibleExchange   (unsigned i1, unsigned k1, unsigned i2, unsigned k2);

     // DP move
         bool     ExploreDP();

     // Extended
        bool     ExploreExtended(bool objective);

     // Utils
        double   myRound(double value) {
            unsigned decimals = 6;
            double factor = pow(10,decimals);
            return round(value*factor)/factor;
        }

        template <class T> void rndShuffle(T &vec) {
            Tengine &engine = vengine[omp_get_thread_num()];
            shuffle(vec.begin(),vec.end(),engine);
        }

        unsigned rndUns(unsigned from, unsigned upto) {
            Tengine &engine = vengine[omp_get_thread_num()];
            uniform_int_distribution<unsigned> rndDist(from,upto);
            return rndDist(engine);
        }

        double rndDbl(double from, double upto) {
            Tengine &engine = vengine[omp_get_thread_num()];
            uniform_real_distribution<double> rndDist(from,upto);
            return rndDist(engine);
        }

};

#endif	/* SOLUTION_H */
