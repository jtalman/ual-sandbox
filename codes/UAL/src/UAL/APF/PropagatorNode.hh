// Library       : UAL
// File          : UAL/APF/PropagatorNode.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_PROPAGATOR_NODE_HH
#define UAL_PROPAGATOR_NODE_HH

#include "UAL/Common/AttributeSet.hh"
#include "UAL/Common/Algorithm.hh"
#include "UAL/Common/Probe.hh"
#include "UAL/Common/RCIPtr.hh"
#include "UAL/SMF/AcceleratorNode.hh"

#include <string>

namespace UAL {

/**  A basis interface of a hierarchical tree of accelerator propagator nodes.*/

  class PropagatorNode : public Algorithm {

  public:

    virtual const char* getType() = 0;

    /** Returns true if this node is composite */
    virtual bool isSequence() = 0;

    /** Returns the first node of the accelerator sector associated with this propagator */
    virtual UAL::AcceleratorNode& getFrontAcceleratorNode() = 0;

    /** Returns the last node of the accelerator sector associated with this propagator */
    virtual UAL::AcceleratorNode& getBackAcceleratorNode() = 0;

    /** Defines the lattice elemements 
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const AcceleratorNode& sequence, int i0, int i1, 
				    const AttributeSet& attSet) = 0;

    /** Propagates probe through the associated accelerator nodes */
    virtual void propagate(UAL::Probe& probe) = 0;

    /** Returns a deep copy of this node */
    virtual PropagatorNode* clone() = 0;

  };

}



#endif
