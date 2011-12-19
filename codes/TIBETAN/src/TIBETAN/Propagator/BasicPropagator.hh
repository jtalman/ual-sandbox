// Library       : TIBETAN
// File          : TIBETAN/Propagator/BasicPropagator.hh
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#ifndef UAL_TIBETAN_BASIC_PROPAGATOR_HH
#define UAL_TIBETAN_BASIC_PROPAGATOR_HH

#include "UAL/APF/PropagatorComponent.hh"
// #include "SMF/PacLattElement.h"

namespace TIBETAN {

  /** A root class of TIBETAN propagators. */

  class BasicPropagator : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BasicPropagator();

    /** Destructor */
    virtual ~BasicPropagator();

    const char* getType();

    /** Returns false */
    bool isSequence() { return false; }

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);


    /** Returns the first node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getBackAcceleratorNode();

    /** Propagates a bunch of particles. */
    void propagate(UAL::Probe& probe);

  private:

    // accelerator name
    std::string m_accName;
    
    // index of the front element
    int m_i0;

    // index of the back element
    int m_i1;

    // front node
    // PacLattElement m_frontNode;
    UAL::AcceleratorNode* m_frontNode;

    // back node
    // PacLattElement m_backNode;
    UAL::AcceleratorNode* m_backNode;
  };

}

#endif
