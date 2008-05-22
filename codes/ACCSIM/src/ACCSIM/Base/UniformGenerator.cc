//# Library       : ACCSIM
//# File          : ACCSIM/Base/UniformGenerator.cc
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#include <math.h>
#include "ACCSIM/Base/UniformGenerator.hh"

// Constructor
ACCSIM::UniformGenerator::UniformGenerator()
{
}

// Returns a random deviate.
double ACCSIM::UniformGenerator::getNumber(int& idum)
{
  return ran(idum);
}




