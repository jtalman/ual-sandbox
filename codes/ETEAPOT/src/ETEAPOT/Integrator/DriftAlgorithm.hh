// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DriftAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_DRIFT_ALGORITHM_HH
#define UAL_ETEAPOT_DRIFT_ALGORITHM_HH

#include "ETEAPOT/Integrator/CommonAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the drift algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DriftAlgorithm 
    : public CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DriftAlgorithm();

    /** Destructor */
    ~DriftAlgorithm();

  };

}

#include "ETEAPOT/Integrator/DriftAlgorithm.icc"

#endif
