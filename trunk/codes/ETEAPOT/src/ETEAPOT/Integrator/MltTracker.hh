// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltTracker.hh
// Copyright     : see Copyright file

#ifndef ETEAPOT_MLT_TRACKER_HH
#define ETEAPOT_MLT_TRACKER_HH

#include <stdlib.h>
#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/MltAlgorithm.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

namespace ETEAPOT {

  /** Multipole Tracker. */

  class MltTracker : public BasicTracker {

  public:

    /** Constructor */
    MltTracker();

    /** Copy constructor */
    MltTracker(const MltTracker& mt);

    /** Destructor */
    ~MltTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);


    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    inline MltData& getMltData();

    static double m_m;

    static int mltK;

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Mlt attributes */
    MltData m_mdata;

    /** Propagator algorithm */
    static MltAlgorithm<double, PAC::Position> s_algorithm;

  private:

    void initialize();
    void copy(const MltTracker& mt);

  };

  inline MltData& MltTracker::getMltData()
  {
      return m_mdata;
  }

}

#endif
