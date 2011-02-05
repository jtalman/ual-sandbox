// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/CommonAlgorithm.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_COMMON_ALGORITHM_HH
#define ETEAPOT_COMMON_ALGORITHM_HH

namespace ETEAPOT {
 
  /** Template of the common methods used by ETEAPOT propagators */

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

#include "ETEAPOT/Integrator/CommonAlgorithm.icc"

#endif
