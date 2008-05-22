//# Library       : ACCSIM
//# File          : ACCSIM/Base/RandomNumberGenerator.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_RANDOM_NUMBER_GENERATOR_HH
#define UAL_ACCSIM_RANDOM_NUMBER_GENERATOR_HH

namespace ACCSIM {

  /** The abstract class of different random generators.*/

  class RandomNumberGenerator 
  {
  public:

    /** Destructor */
    virtual ~RandomNumberGenerator() {};

    /** Returns a random deviate.*/
    virtual double getNumber(int& iseed) = 0;

  protected:

    /** Returns a unform random deviate within [0, 1].
      Initial iseed has to be any negative integer.
    */
    double ran(int& iseed);

  };  

}

#endif
