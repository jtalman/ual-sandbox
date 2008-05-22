#ifndef UAL_AIM_BTF_BASIC_DEVICE_HH
#define UAL_AIM_BTF_BASIC_DEVICE_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorComponent.hh"

namespace AIM {

  /** Basic class of  Beam Transfer Function (BTF) devices */

  class BTFBasicDevice : public UAL::PropagatorNode {

  public:

    /** Constructor */
    BTFBasicDevice() {}

    /** Destructor */
    virtual ~BTFBasicDevice() {}

    /** Returns false */
    bool isSequence() { return false; }

    /** Set lattice elements */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Returns the first node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    UAL::AcceleratorNode& getBackAcceleratorNode();


  private:

    /** front node */
    UAL::AcceleratorNode* m_frontNode;

    /** back node */
    UAL::AcceleratorNode* m_backNode;

  };

}

#endif
