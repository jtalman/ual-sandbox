// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleTracker.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_DIPOLE_TRACKER_HH
#define UAL_ETEAPOT_DIPOLE_TRACKER_HH

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "ETEAPOT/Integrator/DipoleData.hh"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/DipoleAlgorithm.hh"
#include "newDipoleAlgorithm.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#define MAXSXF 1000

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

    inline MltData& getElectricData();

    static double m_m;

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Dipole attributes */
    DipoleData m_data;

    /** Electric attributes */
    MltData m_edata;

    /** Propagator algorithm */
    static newDipoleAlgorithm<double, PAC::Position> s_algorithm;

  };

  inline DipoleData& DipoleTracker::getDipoleData()
  {
      return m_data;
  }

  inline MltData& DipoleTracker::getElectricData()
  {
      return m_edata;
  }

}

#endif
