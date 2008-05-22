//# Library       : ACCSIM
//# File          : ACCSIM/Base/OrbitUniformGenerator.hh
//# Copyright     : see Copyright file
//# Original code : 
//# C++ version   : 

#ifndef UAL_ACCSIM_ORBIT_UNIFORM_GENERATOR_HH
#define UAL_ACCSIM_ORBIT_UNIFORM_GENERATOR_HH

#include "ACCSIM/Base/RandomNumberGenerator.hh"

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

 

namespace ACCSIM {

/** The Orbit UniformGenerator class produces uniform random deviates within [0, 1].  
*/

  class OrbitUniformGenerator : public ACCSIM::RandomNumberGenerator
  {
  public:
  
    /** Constructor */
    OrbitUniformGenerator();

    /** Returns a random deviate. */
    virtual double getNumber(int& iseed);

  protected:

    double ran1(int& iseed);

  };
}

#endif
