//# Library       : ACCSIM
//# File          : ACCSIM/Base/OrbitGaussianGenerator.hh
//# Copyright     : see Copyright file
//# Original code : 
//# C++ version   :  

#ifndef UAL_ACCSIM_ORBIT_GAUSSIAN_GENERATOR_HH
#define UAL_ACCSIM_ORBIT_GAUSSIAN_GENERATOR_HH

#include "ACCSIM/Base/OrbitUniformGenerator.hh"

namespace ACCSIM {

/** The GaussianGenerator class produces normal (Gaussian) 
    deviates. 
*/ 

  class OrbitGaussianGenerator : public OrbitUniformGenerator
  {
  public:
  
    /** Constructor */
    OrbitGaussianGenerator();

    /** Returns a random deviate. */
    virtual double getNumber(int& iseed);

  private:

    int iset_;
    double gset_;

  };

}

#endif
