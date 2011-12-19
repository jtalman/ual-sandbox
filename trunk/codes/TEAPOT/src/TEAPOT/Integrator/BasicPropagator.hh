// Library       : TEAPOT
// File          : TEAPOT/Integrator/BasicPropagator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_BASIC_PROPAGATOR_HH
#define UAL_TEAPOT_BASIC_PROPAGATOR_HH

#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattElement.h"

namespace TEAPOT {

  /** A root class of TEAPOT conventional and DA integrators. */

  class BasicPropagator : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BasicPropagator();

    /** Destructor */
    virtual ~BasicPropagator();

    const char* getType();

    /** Returns fasle */
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


    /** Propagates a probe which can be a bunch (PAC::Bunch) or a Taylor map (PAC::TMap) */
    void propagate(UAL::Probe& probe);

    /** Makes copy */
    UAL::PropagatorNode* clone();

  protected:

    /** Integrator steps */
    static double s_steps[5];

    /** front node */
    PacLattElement m_frontNode;

    /** back node */
    PacLattElement m_backNode;
  
  };

}

#endif
