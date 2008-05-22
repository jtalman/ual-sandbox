// Library       : ICE
// File          : ICE/Base/BasicPropagator.hh
// Copyright     : see Copyright file
// Author        : M.Blaskiewicz
// C++ version   : N.Malitsky 

#ifndef UAL_ICE_BASIC_PROPAGATOR_HH
#define UAL_ICE_BASIC_PROPAGATOR_HH

#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattElement.h"

namespace ICE {

  /** A root class of ICE propagators. */

  class BasicPropagator : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BasicPropagator();

    /** Destructor */
    virtual ~BasicPropagator();

    /** Returns false */
    bool isSequence() { return false; }

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);


    /** Returns the first node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    UAL::AcceleratorNode& getBackAcceleratorNode();


    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

  protected:

    /** front node */
    PacLattElement m_frontNode;

    /** back node */
    PacLattElement m_backNode;
  
  };

}

#endif
