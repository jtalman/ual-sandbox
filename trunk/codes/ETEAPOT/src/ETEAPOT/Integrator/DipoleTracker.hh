// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleTracker.hh
// Copyright     : see Copyright file

#ifndef ETEAPOT_DIPOLE_TRACKER_HH
#define ETEAPOT_DIPOLE_TRACKER_HH

#include "ETEAPOT/Integrator/DipoleData.hh"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/DipoleAlgorithm.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

namespace ETEAPOT {

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

    inline MltData& getMltData();

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Dipole attributes */
    DipoleData m_data;

    /** Electric attributes */
    MltData m_edata;

    /** Propagator algorithm */
    static DipoleAlgorithm<double, PAC::Position> s_algorithm;

  };

  inline DipoleData& DipoleTracker::getDipoleData()
  {
      return m_data;
  }

  inline MltData& DipoleTracker::getMltData()
  {
      return m_edata;
  }

}

#endif
