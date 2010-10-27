// Library       : TEAPOT
// File          : TEAPOT/Integrator/MltTracker.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_MLT_TRACKER_HH
#define UAL_TEAPOT_MLT_TRACKER_HH

#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/MagnetAlgorithm.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"

namespace TEAPOT {

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

    inline MagnetData& getMagnetData();

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Magnet attributes */
    MagnetData m_mdata;

    /** Propagator algorithm */
    static MagnetAlgorithm<double, PAC::Position> s_algorithm;

  private:

    void initialize();
    void copy(const MltTracker& mt);

  };

  inline MagnetData& MltTracker::getMagnetData()
  {
      return m_mdata;
  }

}

#endif
