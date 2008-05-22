//# Library     : ACCSIM
//# File        : ACCSIM/Base/RandomNumberGenerator.cc
//# Copyright   : see Copyright file
//# Author      : F.W.Jones
//# C++ version : N.Malitsky 

#include <stdlib.h>
#include <iostream>
#include "ACCSIM/Base/RandomNumberGenerator.hh"

// Returns a unform random deviate within [0, 1].
// Initial iseed has to be any negative integer.
// Reference: Cambridge U. Press. Numerical Recipes in C. p.212.

double ACCSIM::RandomNumberGenerator::ran(int& idum)
{
  const int  M = 714025;
  const int IA =   1366;
  const int IC = 150889;

  static long iy, ir[98];
  static int iff = 0;
  int j;

  // As a precaution aginst misuse, we will always initialize 
  // on the first call, even if idum is not set negative.
  if(idum < 0 || iff == 0) {
    iff = 1;
    if((idum = (IC - idum % M))  < 0) idum = -idum;
    for(j = 1; j < 97; j++) {
      idum = (IA*idum+IC) % M;
      ir[j] = idum;
    }
    idum = (IA*idum + IC) % M;
    iy = idum;
  }
  j = (int) (1 + 97.0*iy/M);


  // This is where we start if not initializing. Use the 
  // previously saved random number y to get get index j 
  // between 1 and 97. Then use the corresponding ir[j] for 
  // both the next j and the output number.

  if(j > 97 || j < 1) {
    std::cerr << "Error: ACCSIM_BunchGenerator::ran: This cannot happen.\n";
    exit(1);
  }

  iy = ir[j];
  idum = (IA*idum + IC) % M;
  ir[j] = idum;

  // std::cout << idum << std::endl;
  return (double) iy/M;  
}
