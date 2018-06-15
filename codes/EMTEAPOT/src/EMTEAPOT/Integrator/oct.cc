#ifndef EMTEAPOT_OCT_CC
#define EMTEAPOT_OCT_CC

#include <iomanip>
#include <stdlib.h>

#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"
#include "EMTEAPOT/Integrator/genMethods/Matrices.hh"

#include"EMTEAPOT/Integrator/genMethods/Vectors.h"
#include"EMTEAPOT/Integrator/genMethods/spinExtern"
#include"EMTEAPOT/Integrator/genMethods/designExtern"
#include"EMTEAPOT/Integrator/genMethods/bunchParticleExtern"

namespace EMTEAPOT {

  class oct : public ETEAPOT::BasicTracker {

  public:

#include"EMTEAPOT/Integrator/octMethods/classMethods"

    inline ETEAPOT::MltData& getMltData();

#include"EMTEAPOT/Integrator/octMethods/propagate.method"
#include"EMTEAPOT/Integrator/genMethods/get_vlcyMKS.method"
#include"EMTEAPOT/Integrator/genMethods/passDrift.method"

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Mlt attributes */
    ETEAPOT::MltData m_mdata;

  };

  inline ETEAPOT::MltData& oct::getMltData()
  {
      return m_mdata;
  }

}

#endif
