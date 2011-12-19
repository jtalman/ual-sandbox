//# Library       : SIMBAD
//# File          : SIMBAD/Base/BasicPropagator.hh
//# Copyright     : see Copyright file
//# Authors       : N.Malitsky and N.D'Imperio

#ifndef UAL_SIMBAD_BASIC_PROPAGATOR_HH
#define UAL_SIMBAD_BASIC_PROPAGATOR_HH


#include "UAL/APF/PropagatorComponent.hh"


namespace SIMBAD {

  /** A root class of the SIMBAD propagators */

  class BasicPropagator : public UAL::PropagatorNode
  {
  public:

    const char* getType();

    /** Returns false (PropagatorNode method) */
    bool isSequence() { return false; }

    /** Returns the first node of the accelerator sector associated with this propagator
	(PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
	(PropagatorNode method)
     */    
    virtual UAL::AcceleratorNode& getBackAcceleratorNode();

    /** Defines the lattice elemements 
	(PropagatorNode method)
    */    
    virtual void  setLatticeElements (const UAL::AcceleratorNode &, int, int, const UAL::AttributeSet &);

  protected:

    /** front node */
    UAL::AcceleratorNode* m_frontNode;

    /** back node */
    UAL::AcceleratorNode* m_backNode;

    
  };


}



#endif
