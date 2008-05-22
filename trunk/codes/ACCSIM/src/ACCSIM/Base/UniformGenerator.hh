//# Library       : ACCSIM
//# File          : ACCSIM/Base/UniformGenerator.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_UNIFORM_GENERATOR_HH
#define UAL_ACCSIM_UNIFORM_GENERATOR_HH

#include "ACCSIM/Base/RandomNumberGenerator.hh"

namespace ACCSIM {

  /** Random generator for producing uniform random deviates within [0, 1].*/

  class UniformGenerator : public ACCSIM::RandomNumberGenerator
  {
  public:
  
    /** Constructor */
    UniformGenerator();

    /** Returns a random deviate. */
    virtual double getNumber(int& iseed);

  private:

  };
}

#endif
