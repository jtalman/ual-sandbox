// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleTracker.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DIPOLE_TRACKER_HH
#define UAL_TEAPOT_DIPOLE_TRACKER_HH

#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/DipoleAlgorithm.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"

namespace TEAPOT {

  /** Dipole tracker. */

  class DipoleTracker : public BasicTracker {

  public:

    /** Constructor */
    DipoleTracker();

    /** Copy constructor */
    DipoleTracker(const DipoleTracker& dt);

    /** Destructor */
    ~DipoleTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    inline DipoleData& getDipoleData();

    inline MagnetData& getMagnetData();

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Dipole attributes */
    DipoleData m_data;

    /** Magnet attributes */
    MagnetData m_mdata;

    /** Propagator algorithm */
    static DipoleAlgorithm<double, PAC::Position> s_algorithm;

  };

  inline DipoleData& DipoleTracker::getDipoleData()
  {
      return m_data;
  }

  inline MagnetData& DipoleTracker::getMagnetData()
  {
      return m_mdata;
  }

}

#endif
