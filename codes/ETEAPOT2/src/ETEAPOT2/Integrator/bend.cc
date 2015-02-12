#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>

#include <typeinfo> 

#include "SMF/PacLattice.h"

#include "ETEAPOT/Integrator/DipoleData.hh"
#include "ETEAPOT/Integrator/MltData.hh"

//#include "ETEAPOT2/Integrator/BasicTracker.hh"
#include "UAL/APF/PropagatorComponent.hh"

#include "UAL/Common/Def.hh"
#include "SMF/PacLattElement.h"
#include "PAC/Beam/Position.hh"
#include "SMF/PacElemAperture.h"
#include "SMF/PacElemOffset.h"

#include "ETEAPOT2/Integrator/genMethods/Matrices.hh"
#include "ETEAPOT2/Integrator/genMethods/Vectors.h"
#include "ETEAPOT2/Integrator/genMethods/spinExtern"
#include "ETEAPOT2/Integrator/genMethods/designExtern"
#include "ETEAPOT2/Integrator/genMethods/bunchParticleExtern"

namespace ETEAPOT2 {
 class bend : public UAL::PropagatorNode {
// class bend : public ETEAPOT2::BasicPropagator {
// class bend : public ETEAPOT2::BasicTracker {
  public:
#include "ETEAPOT2/Integrator/bendMethods/class.methods"

#include "ETEAPOT2/Integrator/bendMethods/propagate.method"

#include "ETEAPOT2/Integrator/bendMethods/entryFF.method"
#include "ETEAPOT2/Integrator/bendMethods/refractIn.method"
#include "ETEAPOT2/Integrator/bendMethods/traverseSplitBendExactly.method"
#include "ETEAPOT2/Integrator/bendMethods/refractOut.method"
#include "ETEAPOT2/Integrator/bendMethods/exitFF.method"

#include "ETEAPOT2/Integrator/bendMethods/updateDesignParameters.method"
#include "ETEAPOT2/Integrator/bendMethods/munoz.methods"
#include "ETEAPOT2/Integrator/bendMethods/classGlobals"
#include "ETEAPOT2/Integrator/bendMethods/timeViaExpansion"
#include "ETEAPOT2/Integrator/bendMethods/getR.method"
#include "ETEAPOT2/Integrator/bendMethods/getRinverse.method"
#include "ETEAPOT2/Integrator/bendMethods/update_xi.method"
#include "ETEAPOT2/Integrator/bendMethods/updateMunoz.method"
#include "ETEAPOT2/Integrator/bendMethods/initSpin.method"
#include "ETEAPOT2/Integrator/bendMethods/updateSpin.method"
#include "ETEAPOT2/Integrator/bendMethods/get_dr.method"
#include "ETEAPOT2/Integrator/bendMethods/get_rOutV_rot.method"
#include "ETEAPOT2/Integrator/bendMethods/get_pOutV_rot.method"
#include "ETEAPOT2/Integrator/bendMethods/get_dpxBypDc.method"
#include "ETEAPOT2/Integrator/bendMethods/get_dt.method"
#include "ETEAPOT2/Integrator/bendMethods/get_dt_xi.method"
#include "ETEAPOT2/Integrator/bendMethods/perSplitBendOutput.method"
 };
}
