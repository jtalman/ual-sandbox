#ifndef ETEAPOT_DIPOLE_TRACKER_MLT_TURN_HH
#define ETEAPOT_DIPOLE_TRACKER_MLT_TURN_HH

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "ETEAPOT/Integrator/DipoleData.hh"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/algorithm.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#define MAXSXF 1000

namespace ETEAPOT_MltTurn {

  /** bend tracker. */

  class DipoleTracker : public ETEAPOT::BasicTracker {

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

    inline ETEAPOT::DipoleData& getDipoleData();

    inline ETEAPOT::MltData& getElectricData();

    static double m_m;

    static int bend;

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** bend attributes */
    ETEAPOT::DipoleData m_data;

    /** Electric attributes */
    ETEAPOT::MltData m_edata;

    /** Propagator algorithm */
    static algorithm<double, PAC::Position> s_algorithm;

  };

  inline ETEAPOT::DipoleData& DipoleTracker::getDipoleData()
  {
      return m_data;
  }

  inline ETEAPOT::MltData& DipoleTracker::getElectricData()
  {
      return m_edata;
  }

}

#endif
