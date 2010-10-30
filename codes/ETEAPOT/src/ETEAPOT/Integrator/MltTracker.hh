// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltTracker.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_MLT_TRACKER_HH
#define UAL_ETEAPOT_MLT_TRACKER_HH

#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/ElectricData.hh"
#include "ETEAPOT/Integrator/ElectricAlgorithm.hh"
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

    inline ElectricData& getElectricData();

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Electric attributes */
    ElectricData m_edata;

    /** Propagator algorithm */
    static ElectricAlgorithm<double, PAC::Position> s_algorithm;

  private:

    void initialize();
    void copy(const MltTracker& mt);

  };

  inline ElectricData& MltTracker::getElectricData()
  {
      return m_edata;
  }

}

#endif
