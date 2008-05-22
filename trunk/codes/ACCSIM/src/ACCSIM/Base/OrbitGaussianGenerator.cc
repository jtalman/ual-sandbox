//# Library       : ACCSIM
//# File          : ACCSIM/Base/OrbitGaussianGenerator.cc
//# Copyright     : see Copyright file
//# Original code : 
//# C++ version   : 

#include <iostream>
#include <math.h>
#include "ACCSIM/Base/OrbitGaussianGenerator.hh"

// Constructor

ACCSIM::OrbitGaussianGenerator::OrbitGaussianGenerator()
  : iset_(0), gset_(0.0)
{
}

// Returns a random deviate.
// Reference: Cambridge U. Press. Numerical Recipes in C. p.217.

double ACCSIM::OrbitGaussianGenerator::getNumber(int& idum)
{
  float fac, r, v1, v2;
  int counter = 0;
  if(iset_ == 0){
    do {
      counter++;
      v1 = 2.0*ran1(idum) - 1.0; // pick two uniform numbers in the square 
      v2 = 2.0*ran1(idum) - 1.0; // extending from -1 to +1 in each direction
   
      r = v1*v1 + v2*v2;       // see if they are in the unit circle
    } while(r > 1.0);          // and if they are not, try again

    fac = sqrt(-2.0*log(r)/r);
    // std::cout << "v1 = " << v1 << " v2 = " << v2 << " fac= " << fac << " counter= " << counter << endl;

    // Now make the Box-Muller transformation to get two normal deviates.
    // Return one and save the other for next time.

    gset_ = v1*fac;
    iset_ = 1;       // Set flag
    return v2*fac;
  } else {           // We have an extra deviate handy, 
    iset_ = 0;       // so unset the flag,
    return gset_;    // and return it
  }

}


