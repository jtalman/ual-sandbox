// Library       :
// File          : 
// Copyright     : see Copyright file
// Author        :
// C++ version   : N.Malitsky and J.Talman

#ifndef THINSPIN_SPIN_COMMON_ALGORITHM_HH
#define THINSPIN_SPIN_COMMON_ALGORITHM_HH

#include "extern_globalBlock.cc"

namespace THINSPIN {
 
  /** Template of the common methods used by THINSPIN propagators */

  template<class Coordinate, class Coordinates> class commonAlgorithm {

  public:

    /** Constructor */
    commonAlgorithm();

    /** Destructor */
    ~commonAlgorithm();

    /** Propagates a probe coordinates through a drift*/
    void passDrift(double l, Coordinates& p, Coordinates& tmp, double v0byc);

    void makeVelocity(Coordinates& p, Coordinates& tmp, double v0byc);
    void makeRV(Coordinates& p, Coordinates& tmp, double e0, double p0, double m0);

  };

}

#include "commonAlgorithm.icc"

#endif
