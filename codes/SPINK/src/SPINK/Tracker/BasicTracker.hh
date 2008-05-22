// Library       : SPINK
// File          : SPINK/Tracker/BasicTracker.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_BASIC_TRACKER_HH
#define UAL_SPINK_BASIC_TRACKER_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattElement.h"

namespace SPINK {

  /** A root class of SPINK conventional integrators. */

  class BasicTracker : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BasicTracker();

    /** Destructor */
    virtual ~BasicTracker();

    /** Returns false */
    bool isSequence() { return false; }
    
  };

}

#endif
