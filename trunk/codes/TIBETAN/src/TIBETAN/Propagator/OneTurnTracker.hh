// Library       : TIBETAN
// File          : TIBETAN/Propagator/OneTurnTracker.hh
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 


#ifndef UAL_TIBETAN_ONE_TURN_TRACKER_HH
#define UAL_TIBETAN_ONE_TURN_TRACKER_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorComponent.hh"
#include "SMF/PacLattice.h"
#include "Optics/PacTMap.h"
#include "Main/Teapot.h"
#include "TIBETAN/Propagator/BasicPropagator.hh"

namespace TIBETAN {

  /** One-turn tracker. */

  class OneTurnTracker : public BasicPropagator {

  public:

    /** Constructor */
    OneTurnTracker();

    /** Copy Constructor */
    OneTurnTracker(const OneTurnTracker& st);

    /** Destructor */
    ~OneTurnTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates bunch (inherited from UAL::PropagatorNode) */
    void propagate(UAL::Probe& probe);

  protected:

    // static PacLattice s_lattice;
    // static Teapot     s_teapot;

    double m_suml, m_alpha0;

    double m_mux, m_betax, m_alphax, m_dx, m_dpx;
    double m_muy, m_betay, m_alphay, m_dy, m_dpy;

    double m_chromx;
    double m_chromy;

  private:

    void init();
    void copy(const OneTurnTracker& st);

  };

  class OneTurnTrackerRegister 
  {
    public:

    OneTurnTrackerRegister(); 
  };

  

}

#endif
