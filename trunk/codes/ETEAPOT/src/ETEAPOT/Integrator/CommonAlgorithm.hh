// Library       : TEAPOT
// File          : TEAPOT/Integrator/CommonAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_COMMON_ALGORITHM_HH
#define UAL_TEAPOT_COMMON_ALGORITHM_HH

namespace TEAPOT {
 
  /** Template of the common methods used by TEAPOT propagators */

  template<class Coordinate, class Coordinates> class CommonAlgorithm {

  public:

    /** Constructor */
    CommonAlgorithm();

    /** Destructor */
    ~CommonAlgorithm();

    /** Propagates a probe coordinates through a drift*/
    void passDrift(double l, Coordinates& p, Coordinates& tmp, double v0byc);

    void makeVelocity(Coordinates& p, Coordinates& tmp, double v0byc);
    void makeRV(Coordinates& p, Coordinates& tmp, double e0, double p0, double m0);

  };

}

#include "TEAPOT/Integrator/CommonAlgorithm.icc"

#endif
