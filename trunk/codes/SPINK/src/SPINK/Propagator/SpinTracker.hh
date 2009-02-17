// Library       : SPINK
// File          : SPINK/Propagator/SpinTracker.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_SPIN_TRACKER_HH
#define UAL_SPINK_SPIN_TRACKER_HH

#include "SPINK/Propagator/SpinPropagator.hh"

namespace SPINK {

  /** Basis class of different spin mappers */

  class SpinTracker : public SpinPropagator {

  public:

    /** Constructor */
    SpinTracker();

    /** Destructor */
    ~SpinTracker();

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

  protected:

    /** Conventional tracker */
    UAL::PropagatorNodePtr m_tracker;

    /** Buffer of positions used internally by spin trackers */
    static std::vector<PAC::Position> s_positions;

  };


}

#endif
