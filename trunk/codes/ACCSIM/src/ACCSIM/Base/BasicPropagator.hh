//# Library       : ACCSIM
//# File          : ACCSIM/Base/BasicPropagator.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky

#ifndef UAL_ACCSIM_BASIC_PROPAGATOR_HH
#define UAL_ACCSIM_BASIC_PROPAGATOR_HH


#include "UAL/APF/PropagatorComponent.hh"





namespace ACCSIM {

  /** A root class of the ACCSIM propagators */

  class BasicPropagator : public UAL::PropagatorNode
  {
  public:

    /** Constructor */
    BasicPropagator();

    /** Destructor */
    virtual ~BasicPropagator();

    const char* getType();

    /** Returns false */
    bool isSequence() { return false; }

    /** Propagates  a bunch */
    void propagate(UAL::Probe& bunch);

    virtual UAL::AcceleratorNode& getFrontAcceleratorNode();
    
    virtual UAL::AcceleratorNode& getBackAcceleratorNode();
    
    virtual void  setLatticeElements (const UAL::AcceleratorNode &, int, int, const UAL::AttributeSet &);
    
    UAL::PropagatorNode* clone ();
  
  private:
    /** front node */
    UAL::AcceleratorNode* m_frontNode;

    /** back node */
    UAL::AcceleratorNode* m_backNode;

    
  };


}



#endif
