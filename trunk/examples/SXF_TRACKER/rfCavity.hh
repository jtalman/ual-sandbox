// Library       : SXF_TRACKER
// File          : examples/SXF_TRACKER/rfCavity.hh
// Copyright     : see Copyright file
// Author        : 
// C++ version   : J.Talman, N.Malitsky

#ifndef UAL_RFCAVITY_HH
#define UAL_RFCAVITY_HH

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "TEAPOT/Integrator/BasicTracker.hh"

namespace SXF_TRACKER {

  /** SXF_Tracker-specific RF Cavity Tracker */

  class rfCavity : public TEAPOT::BasicTracker {

  public:

    /** Constructor */
    rfCavity();

    /** Copy Constructor */
    rfCavity(const rfCavity& rft);

    /** Destructor */
    virtual ~rfCavity();

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
    void copy(const rfCavity& rft);
    
  };

  class rfCavityRegister
  {
    public:

      rfCavityRegister();
  };

}

#endif
