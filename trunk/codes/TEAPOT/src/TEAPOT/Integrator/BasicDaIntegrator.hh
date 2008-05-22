// Library       : TEAPOT
// File          : TEAPOT/Integrator/BasicDaIntegrator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_BASIC_DA_INTEGRATOR_HH
#define UAL_TEAPOT_BASIC_DA_INTEGRATOR_HH

#include "SMF/PacLattElement.h"
#include "PAC/Beam/Position.hh"

#include "TEAPOT/Integrator/BasicPropagator.hh"

namespace TEAPOT {

  /** A root class of TEAPOT DA integrators. */

  class BasicDaIntegrator : public TEAPOT::BasicPropagator {

  public:

    /** Constructor */
    BasicDaIntegrator();

    /** Destructor */
    ~BasicDaIntegrator();

    
  };

}

#endif
