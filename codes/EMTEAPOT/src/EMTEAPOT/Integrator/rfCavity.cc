#ifndef UAL_EMTEAPOT_RFCAVITY_CC
#define UAL_EMTEAPOT_RFCAVITY_CC

#include<iomanip>

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#include"EMTEAPOT/Integrator/genMethods/Vectors.h"
#include"EMTEAPOT/Integrator/genMethods/spinExtern"
#include"EMTEAPOT/Integrator/genMethods/designExtern"
#include"EMTEAPOT/Integrator/genMethods/bunchParticleExtern"

namespace EMTEAPOT {

  /** RF Cavity Tracker */

  class rfCavity : public ETEAPOT::BasicTracker {

  public:

#include"EMTEAPOT/Integrator/rfCavityMethods/classMethods"

#include"EMTEAPOT/Integrator/rfCavityMethods/propagate.method"
#include"EMTEAPOT/Integrator/genMethods/get_vlcyMKS.method"
#include"EMTEAPOT/Integrator/genMethods/passDrift.method"

   double m_V;
   double m_lag;
   double m_h;

  };

}

#endif
