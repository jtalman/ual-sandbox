#ifndef ETEAPOT2_DRIFT_CC
#define ETEAPOT2_DRIFT_CC

#include<iomanip>
#include<cstdlib>

#include<math.h>

#include"PAC/Beam/Bunch.hh"
#include"SMF/PacLattice.h"

#include"ETEAPOT/Integrator/BasicTracker.hh"

#include"ETEAPOT2/Integrator/genMethods/Vectors.h"
#include"ETEAPOT2/Integrator/genMethods/spinExtern"
#include"ETEAPOT2/Integrator/genMethods/designExtern"
#include"ETEAPOT2/Integrator/genMethods/bunchParticleExtern"

namespace ETEAPOT2 {

 class drift : public ETEAPOT::BasicTracker {

  public:

   #include"ETEAPOT2/Integrator/driftMethods/classMethods"

   #include"ETEAPOT2/Integrator/driftMethods/propagate.method"
   #include"ETEAPOT2/Integrator/genMethods/get_vlcyMKS.method"
   #include"ETEAPOT2/Integrator/genMethods/passDrift.method"

 };

}

#endif
