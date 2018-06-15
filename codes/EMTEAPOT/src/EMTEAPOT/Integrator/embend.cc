#ifndef EMTEAPOT_EMBEND_CC
#define EMTEAPOT_EMBEND_CC

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>

#include <typeinfo> 

#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/MltData.hh"

#include "ETEAPOT/Integrator/DipoleData.hh"

#include "UAL/APF/PropagatorComponent.hh"

#include "UAL/Common/Def.hh"
#include "SMF/PacLattElement.h"
#include "PAC/Beam/Position.hh"
#include "SMF/PacElemAperture.h"
#include "SMF/PacElemOffset.h"

#include "EMTEAPOT/Integrator/genMethods/Matrices.hh"
#include "EMTEAPOT/Integrator/genMethods/Vectors.h"
#include "EMTEAPOT/Integrator/genMethods/spinExtern"
#include "EMTEAPOT/Integrator/genMethods/designExtern"
#include "EMTEAPOT/Integrator/genMethods/bunchParticleExtern"

namespace EMTEAPOT {
 class embend : public UAL::PropagatorNode {
  public:
#include "EMTEAPOT/Integrator/embendMethods/class.methods"

inline ETEAPOT::MltData& getMltData();

#include "EMTEAPOT/Integrator/embendMethods/propagate.method"

#include "EMTEAPOT/Integrator/embendMethods/entryFF.method"
#include "EMTEAPOT/Integrator/embendMethods/refractIn.method"
#include "EMTEAPOT/Integrator/embendMethods/traverseSplitBendExactly.method"
#include "EMTEAPOT/Integrator/embendMethods/refractOut.method"
#include "EMTEAPOT/Integrator/embendMethods/exitFF.method"

#include "EMTEAPOT/Integrator/embendMethods/updateDesignParameters.method"
#include "EMTEAPOT/Integrator/embendMethods/munoz.methods"
#include "EMTEAPOT/Integrator/embendMethods/classGlobals"
#include "EMTEAPOT/Integrator/embendMethods/timeViaExpansion"
#include "EMTEAPOT/Integrator/embendMethods/getR.method"
#include "EMTEAPOT/Integrator/embendMethods/getRinverse.method"
#include "EMTEAPOT/Integrator/embendMethods/update_xi.method"
#include "EMTEAPOT/Integrator/embendMethods/updateMunoz.method"
#include "EMTEAPOT/Integrator/embendMethods/initSpin.method"
#include "EMTEAPOT/Integrator/embendMethods/updateSpin.method"
#include "EMTEAPOT/Integrator/embendMethods/get_dr.method"
#include "EMTEAPOT/Integrator/embendMethods/get_rOutV_rot.method"
#include "EMTEAPOT/Integrator/embendMethods/get_pOutV_rot.method"
#include "EMTEAPOT/Integrator/embendMethods/get_dpxBypDc.method"
#include "EMTEAPOT/Integrator/embendMethods/get_dt.method"
#include "EMTEAPOT/Integrator/embendMethods/get_dt_xi.method"
#include "EMTEAPOT/Integrator/embendMethods/perSplitBendOutput.method"

/** Element length */
// double m_l;

/** Complexity number */
   double m_ir;

/** Mlt attributes */
   ETEAPOT::MltData m_mdata;

};  

inline ETEAPOT::MltData& embend::getMltData()
{
 return m_mdata;
}

}

#endif
