// Library       : TEAPOT
// File          : TEAPOT/Integrator/DriftAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DRIFT_ALGORITHM_HH
#define UAL_TEAPOT_DRIFT_ALGORITHM_HH

#include "TEAPOT/Integrator/CommonAlgorithm.hh"

namespace TEAPOT {
 
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

#include "TEAPOT/Integrator/DriftAlgorithm.icc"

#endif
