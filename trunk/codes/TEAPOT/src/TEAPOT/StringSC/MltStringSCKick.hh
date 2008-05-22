#ifndef UAL_TEAPOT_MLT_STRING_SCKICK_HH
#define UAL_TEAPOT_MLT_STRING_SCKICK_HH

#include "TEAPOT/Integrator/BasicTracker.hh"
#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/MagnetAlgorithm.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"

namespace TEAPOT {

  /** Mlt tracker. */

  class MltStringSCKick : public BasicTracker {

  public:

    /** Constructor */
    MltStringSCKick();

    /** Copy constructor */
    MltStringSCKick(const MltStringSCKick& dt);

    /** Destructor */
    ~MltStringSCKick();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

  protected:

    // Sets the lattice element 
    void setLatticeElement(const PacLattElement& e);
    void propagateSimpleElement(PAC::Bunch& bunch, double v0byc);
    void propagateComplexElement(PAC::Bunch& bunch, double v0byc);

  protected:

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Magnet attributes */
    MagnetData m_mdata;

    /** Propagator algorithm */
    static MagnetAlgorithm<double, PAC::Position> s_algorithm;

  private:

    void initialize();
    void copy(const MltStringSCKick& mt);


  };

  class MltStringSCKickRegister 
  {
    public:

    MltStringSCKickRegister(); 
  };



}



#endif
