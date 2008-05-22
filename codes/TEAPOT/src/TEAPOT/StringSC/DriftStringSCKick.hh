
#ifndef UAL_TEAPOT_DRIFT_STRING_SCKICK_HH
#define UAL_TEAPOT_DRIFT_STRING_SCKICK_HH

#include "TEAPOT/Integrator/BasicTracker.hh"
#include "TEAPOT/Integrator/DriftAlgorithm.hh"

namespace TEAPOT {

  /** Drift tracker. */

  class DriftStringSCKick : public BasicTracker {

  public:

    /** Constructor */
    DriftStringSCKick();

    /** Copy constructor */
    DriftStringSCKick(const DriftStringSCKick& dt);

    /** Destructor */
    ~DriftStringSCKick();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

  protected:

    // Sets the lattice element 
    // void setLatticeElement(const PacLattElement& e);

  protected:

    /** Propagator algorithm */
    static DriftAlgorithm<double, PAC::Position> s_algorithm;

  private:

    // void initialize();
    // void copy(const DriftTracker& dt);

  };

  class DriftStringSCKickRegister 
  {
    public:

    DriftStringSCKickRegister(); 
  };



}



#endif
