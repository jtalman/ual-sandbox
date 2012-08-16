// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MarkerTracker.hh
// Copyright     : see Copyright file

#ifndef ETEAPOT_MARKER_TRACKER_H
#define ETEAPOT_MARKER_TRACKER_HH

#include "ETEAPOT/Integrator/BasicTracker.hh"

namespace ETEAPOT {

  /** Marker tracker. */

  class MarkerTracker : public BasicTracker {

  public:

    /** Constructor */
    MarkerTracker();

    /** Copy constructor */
    MarkerTracker(const MarkerTracker& dt);

    /** Destructor */
    ~MarkerTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    static int mark;
     static std::string Mark_m_elementName[1000];
     static double Mark_m_sX[1000];

  protected:

    // Sets the lattice element 
    // void setLatticeElement(const PacLattElement& e);

  protected:

    /** Propagator algorithm */
//  static MarkerAlgorithm<double, PAC::Position> s_algorithm;

  private:

    // void initialize();
    // void copy(const MarkerTracker& dt);

  };

}

#endif
