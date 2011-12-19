// Library       : UAL
// File          : UAL/SPF/PropagatorComponent.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_PROPAGATOR_COMPONENT_HH
#define UAL_PROPAGATOR_COMPONENT_HH

#include "UAL/APF/PropagatorNode.hh"

namespace UAL {

  /** A basis interface of propagator nodes, association of accelerator sectors and algorithms.
   */

  class PropagatorComponent : public PropagatorNode {

  public:

    const char* getType();

    /** Returns false */
    bool isSequence();

  };

}

#endif
