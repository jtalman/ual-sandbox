// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DriftTracker.hh
// Copyright     : see Copyright file

#ifndef ETEAPOT_DRIFT_TRACKER_HH
#define ETEAPOT_DRIFT_TRACKER_HH

#include "ETEAPOT/Integrator/BasicTracker.hh"
#include "ETEAPOT/Integrator/DriftAlgorithm.hh"

namespace ETEAPOT {

  /** Drift tracker. */

  class DriftTracker : public BasicTracker {

  public:

    /** Constructor */
    DriftTracker();

    /** Copy constructor */
    DriftTracker(const DriftTracker& dt);

    /** Destructor */
    ~DriftTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    static int drft;

  protected:

    // Sets the lattice element 
    // void setLatticeElement(const PacLattElement& e);

  protected:

    /** Propagator algorithm */
    static DriftAlgorithm<double, PAC::Position> s_algorithm;

  private:

    // void initialize();
    // void copy(const DriftTracker& dt);

  };

}

#endif
