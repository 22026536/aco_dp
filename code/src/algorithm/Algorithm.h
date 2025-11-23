/* 
 * File:   Header file for class Algorithm
 * Author: A. Herrán
 */

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Island.h"

class Algorithm {
    
    public:

     // Constructor/destructor
        Algorithm(const string& pathInstance, unsigned ne);
        virtual ~Algorithm();

     // Public attributes
        vecDbl vtcpu;
        matDbl mcost;

     // Public functions
        void     Iterate();
        void     Migrate();
        Solution GetBest(unsigned i);
        Solution GetBest();
        
        void PrintLogs(double tcpu, unsigned it);
        void SaveLogs();

        static vector<string> Split(const string& str, char delimiter);

    private:
        
     // Private attributes
        Island **islands;

        ofstream fileEvol;
        ofstream fileCost;
        ofstream fileSolu;

     // Private functions
        void Ring  ();
        void Fully (); 
        void Master(); 
        
};

#endif /* ALGORITHM_H */

