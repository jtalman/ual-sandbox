// Library       : SIMBAD
// File          : SIMBAD/Tracker/BasicTracker.hh
// Copyright     : see Copyright file


#ifndef UAL_SIMBAD_BASIC_TRACKER_HH
#define UAL_SIMBAD_BASIC_TRACKER_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattElement.h"

namespace SIMBAD {

  /** A root class of SIMBAD conventional integrators. */

  class BasicTracker : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BasicTracker();

    /** Destructor */
    virtual ~BasicTracker();

    const char* getType();

    /** Returns false */
    bool isSequence() { return false; }
    
  };

}

#endif
