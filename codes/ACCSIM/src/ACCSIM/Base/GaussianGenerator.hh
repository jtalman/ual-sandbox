//# Library       : ACCSIM
//# File          : ACCSIM/Base/GaussianGenerator.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_GAUSSIAN_GENERATOR_HH
#define UAL_ACCSIM_GAUSSIAN_GENERATOR_HH

#include "ACCSIM/Base/RandomNumberGenerator.hh"

namespace ACCSIM {

  /** Random generator for producing normal (Gaussian) deviates. */ 

  class GaussianGenerator : public RandomNumberGenerator
  {
  public:
  
    /** Constructor */
    GaussianGenerator();

    /** Returns a random deviate. */
    virtual double getNumber(int& iseed);

  private:

    int iset_;
    double gset_;

  };

}

#endif
