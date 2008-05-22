// Library       : TIBETAN
// File          : TIBETAN/Propagator/RFCavityTracker.hh
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#ifndef UAL_TIBETAN_RFCAVITY_TRACKER_HH
#define UAL_TIBETAN_RFCAVITY_TRACKER_HH

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "TIBETAN/Propagator/BasicPropagator.hh"

namespace TIBETAN {

  /** RF Cavity Tracker */

  class RFCavityTracker : public BasicPropagator {

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

    double getV() const { return m_V; }
    double getHarmon() const { return m_h; }
    double getLag() const { return m_lag; }

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

  class RfCavityTrackerRegister 
  {
    public:

    RfCavityTrackerRegister(); 
  };


}

#endif
