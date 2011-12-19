// Library       : SPINK
// File          : SPINK/Propagator/SpinPropagator.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_SPIN_PROPAGATOR_HH
#define UAL_SPINK_SPIN_PROPAGATOR_HH

#include "UAL/APF/PropagatorNodePtr.hh"
#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattElement.h"

namespace SPINK {

  /** Basis class of different spin propagators (mappers, integrators, etc.) */

  class SpinPropagator : public UAL::PropagatorNode {

  public:

    /** Constructor */
    SpinPropagator();

    /** Destructor */
    virtual ~SpinPropagator();

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

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

    /** One Turn Spin Matrix  AULNLD:11DEC09 */
    static double OTs_mat[3][3] ;
    //static a1[8] ;

  protected:

    virtual void propagateSpin(UAL::Probe& bunch) = 0;

  protected:

    /**  Front node */
    PacLattElement m_frontNode;

    /** Back node */
    PacLattElement m_backNode;

  };


}

#endif
