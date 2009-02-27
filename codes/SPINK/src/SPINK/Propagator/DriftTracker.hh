
// Library       : SPINK
// File          : SPINK/Propagator/DriftTracker.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky

#ifndef UAL_SPINK_DRIFT_TRACKER_HH
#define UAL_SPINK_DRIFT_TRACKER_HH

#include "SPINK/Propagator/SpinTracker.hh"

namespace SPINK {

  /** Drift tracker */

  class DriftTracker : public SpinTracker {

  public:

    /** Constructor */
    DriftTracker();

   /** Copy constructor */
    DriftTracker(const DriftTracker& st);

    /** Destructor */
    ~DriftTracker();

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1,
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

  protected:


  };

  class DriftTrackerRegister
  {
    public:

    DriftTrackerRegister();
  };


}

#endif


