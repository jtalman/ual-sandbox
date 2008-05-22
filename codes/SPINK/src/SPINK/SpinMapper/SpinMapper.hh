// Library       : SPINK
// File          : SPINK/SpinMapper/SpinMapper.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_SPIN_MAPPER_HH
#define UAL_SPINK_SPIN_MAPPER_HH

#include "SPINK/SpinMapper/SpinPropagator.hh"

namespace SPINK {

  /** Basis class of different spin mappers */

  class SpinMapper : public SpinPropagator {

  public:

    /** Constructor */
    SpinMapper();

    /** Destructor */
    ~SpinMapper();

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

    /** Buffer of positions used internally by spin mappers */
    static std::vector<PAC::Position> s_positions;

  };


}

#endif
