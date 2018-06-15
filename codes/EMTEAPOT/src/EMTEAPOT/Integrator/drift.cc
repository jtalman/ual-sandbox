#ifndef EMTEAPOT_DRIFT_CC
#define EMTEAPOT_DRIFT_CC

#include<iomanip>
#include<cstdlib>

#include<math.h>

#include"PAC/Beam/Bunch.hh"
#include"SMF/PacLattice.h"

#include"ETEAPOT/Integrator/BasicTracker.hh"

#include"EMTEAPOT/Integrator/genMethods/Vectors.h"
#include"EMTEAPOT/Integrator/genMethods/spinExtern"
#include"EMTEAPOT/Integrator/genMethods/designExtern"
#include"EMTEAPOT/Integrator/genMethods/bunchParticleExtern"

namespace EMTEAPOT {

 class drift : public ETEAPOT::BasicTracker {

  public:

   #include"EMTEAPOT/Integrator/driftMethods/classMethods"

   #include"EMTEAPOT/Integrator/driftMethods/propagate.method"
   #include"EMTEAPOT/Integrator/genMethods/get_vlcyMKS.method"
   #include"EMTEAPOT/Integrator/genMethods/passDrift.method"

 };

}

#endif
