
#ifndef UAL_SPINK_RFCAVITY_TRACKER_HH
#define UAL_SPINK_RFCAVITY_TRACKER_HH

#include "PAC/Beam/Position.hh"
#include "SMF/PacLattElement.h"
#include "SMF/PacElemRfCavity.h"
#include "TEAPOT/Integrator/BasicTracker.hh"

#include "SPINK/Propagator/SpinPropagator.hh" //AUL:27APR10

namespace SPINK {

  /** Spink-specific RF Cavity Tracker */

  class RFCavityTracker : public TEAPOT::BasicTracker {

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

    //virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
    //				    const UAL::AttributeSet& attSet); //AUL:27APR10

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    /** Sets Rf parameters */
    static void setRF(double V, double harmon, double lag) { 
      m_V = V; m_h = harmon; m_lag = lag;
    }

    /** Pass ring length AUL:17MAR10 */
    static void setCircum(double circum){circ = circum;}
    static double circ ;

     /** Setup a dump flag for diagnostics AUL:27APR10 */
    static void setOutputDump(bool outdmp){coutdmp = outdmp;}
    static bool coutdmp; 

     /** Pass information on turn number for diagnostics AUL:27APR10 */
    static void setNturns(int iturn){nturn = iturn;}
    static int nturn ;
  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

    /** Propagates the particle through a drift */
    void passDrift(double l, PAC::Position& p, double v0byc, double vbyc);

  protected:

    /** Element length */
    double m_l;

    /** Peak RF voltage [GeV] */ 
    static double m_V;

    /** Phase lag in multiples of 2 pi */
    static double m_lag;

    /** Harmonic number */
    static double m_h;

  private:

    void init();
    void copy(const RFCavityTracker& rft);
    
  };

  class RFCavityTrackerRegister
  {
    public:

      RFCavityTrackerRegister();
  };

}

#endif
