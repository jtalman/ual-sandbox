// Library       : TEAPOT
// File          : TEAPOT/Integrator/DaIntegratorFactory.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DA_INTEGRATOR_FACTORY_HH
#define UAL_TEAPOT_DA_INTEGRATOR_FACTORY_HH

#include "TEAPOT/Integrator/DriftDaIntegrator.hh"
#include "TEAPOT/Integrator/DipoleDaIntegrator.hh"
#include "TEAPOT/Integrator/MltDaIntegrator.hh"
#include "TEAPOT/Integrator/MapDaIntegrator.hh"

namespace TEAPOT {

  /** Factory of the TEAPOT DA integrators */

  class DaIntegratorFactory {

  public:

    /** Returns the DA integrator specified by the element type */
    // static BasicDaIntegrator* createDaIntegrator(const std::string& type);

    /** Returns the default DA integrator */
    static BasicDaIntegrator* createDefaultDaIntegrator();

    /** Returns the drift DA integrator */
    static DriftDaIntegrator* createDriftDaIntegrator();

    /** Returns the dipole DA integrator */
    static DipoleDaIntegrator* createDipoleDaIntegrator();

    /** Returns the multipole DA integrator */
    static MltDaIntegrator* createMltDaIntegrator();

    /** Returns the map evolver */
    static MapDaIntegrator* createMapDaIntegrator();

  };

  class DaIntegratorRegister 
  {
    public:

    DaIntegratorRegister(); 
  };


}

#endif
