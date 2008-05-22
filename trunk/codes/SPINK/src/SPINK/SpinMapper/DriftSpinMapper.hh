// Library       : SPINK
// File          : SPINK/SpinMapper/DriftSpinMapper.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_DRIFT_SPIN_MAPPER_HH
#define UAL_SPINK_DRIFT_SPIN_MAPPER_HH

#include "SPINK/SpinMapper/SpinMapper.hh"

namespace SPINK {

  /** Drift spin mapper */

  class DriftSpinMapper : public SpinMapper {

  public:

    /** Constructor */
    DriftSpinMapper();

    /** Copy constructor */
    DriftSpinMapper(const DriftSpinMapper& sm);

    /** Destructor */
    ~DriftSpinMapper();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

  private:

    void copy(const DriftSpinMapper& sm);

  };


}

#endif
