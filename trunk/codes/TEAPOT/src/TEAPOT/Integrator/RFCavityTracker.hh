// Library       : TEAPOT
// File          : TEAPOT/Integrator/RFCavityTracker.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_RFCAVITY_TRACKER_HH
#define UAL_TEAPOT_RFCAVITY_TRACKER_HH

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "TEAPOT/Integrator/BasicTracker.hh"

namespace TEAPOT {

  /** RF Cavity Tracker */

  class RFCavityTracker : public BasicTracker {

  public:

    /** Constructor */
    RFCavityTracker();

    /** Copy Constructor */
    RFCavityTracker(const RFCavityTracker& rft);

    /** Destructor */
    virtual ~RFCavityTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    /** Sets Rf patameters */
    void setRF(double V, double harmon, double lag);

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

    /** Propagates the particle through a drift */
    void passDrift(double l, PAC::Position& p, double v0byc, double vbyc);

  protected:

    /** Element length */
    double m_l;

    /** Peak RF voltage [GeV] */ 
    double m_V;

    /** Phase lag in multiples of 2 pi */
    double m_lag;

    /** Harmonic number */
    double m_h;

  private:

    void init();
    void copy(const RFCavityTracker& rft);
    
  };

}

#endif
