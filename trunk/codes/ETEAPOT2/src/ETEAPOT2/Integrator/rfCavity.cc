#ifndef UAL_ETEAPOT2_RFCAVITY_CC
#define UAL_ETEAPOT2_RFCAVITY_CC

#include<iomanip>

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#include"ETEAPOT2/Integrator/genMethods/Vectors.h"
#include"ETEAPOT2/Integrator/genMethods/spinExtern"
#include"ETEAPOT2/Integrator/genMethods/designExtern"
#include"ETEAPOT2/Integrator/genMethods/bunchParticleExtern"

namespace ETEAPOT2 {

  /** RF Cavity Tracker */

  class rfCavity : public ETEAPOT::BasicTracker {

  public:

#include"ETEAPOT2/Integrator/rfCavityMethods/classMethods"

#include"ETEAPOT2/Integrator/rfCavityMethods/propagate.method"
#include"ETEAPOT2/Integrator/genMethods/get_vlcyMKS.method"
#include"ETEAPOT2/Integrator/genMethods/passDrift.method"

   double m_V;
   double m_lag;
   double m_h;

  };

}

#endif
