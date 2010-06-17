// Library       : TEAPOT
// File          : TEAPOT/Integrator/bendAndMultipole.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_THINSPIN_BEND_AND_MULTIPOLE_HH
#define UAL_THINSPIN_BEND_AND_MULTIPOLE_HH

#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
//#include "TEAPOT/Integrator/DipoleAlgorithm.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"

#include "bendAndMultipoleAlgorithm.hh"
#include "lorentzTransformForBend.cc"

namespace THINSPIN {

  /** Dipole tracker. */

  class bendAndMultipole : public TEAPOT::BasicTracker {

  public:

    /** Constructor */
    bendAndMultipole();

    /** Copy constructor */
    bendAndMultipole(const bendAndMultipole& dt);

    /** Destructor */
    ~bendAndMultipole();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    inline TEAPOT::DipoleData& getDipoleData();

    inline TEAPOT::MagnetData& getMagnetData();

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Dipole attributes */
    TEAPOT::DipoleData m_data;

    /** Magnet attributes */
    TEAPOT::MagnetData m_mdata;

    /** Propagator algorithm */
    static bendAndMultipoleAlgorithm<double, PAC::Position> s_algorithm;

  };

  class bendAndMultipoleRegister
  {
    public: 

    bendAndMultipoleRegister();
  };

  inline TEAPOT::DipoleData& bendAndMultipole::getDipoleData()
  {
      return m_data;
  }

  inline TEAPOT::MagnetData& bendAndMultipole::getMagnetData()
  {
      return m_mdata;
  }

}

#endif
