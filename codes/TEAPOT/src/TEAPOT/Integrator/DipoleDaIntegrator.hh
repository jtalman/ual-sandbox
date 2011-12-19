// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleDaIntegrator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DIPOLE_DA_INTEGRATOR_HH
#define UAL_TEAPOT_DIPOLE_DA_INTEGRATOR_HH

#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/DipoleAlgorithm.hh"
#include "TEAPOT/Integrator/BasicDaIntegrator.hh"

namespace TEAPOT {

  /** Dipole DA integrator. */

  class DipoleDaIntegrator : public BasicDaIntegrator {

  public:

    /** Constructor */
    DipoleDaIntegrator();

    
    /** Copy constructor */
    DipoleDaIntegrator(const DipoleDaIntegrator& dt);

    /** Destructor */
    ~DipoleDaIntegrator();

    virtual const char* getType();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a vector of truncated power series */
    void propagate(UAL::Probe& probe);

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Dipole attributes */
    DipoleData m_data;

    /** Magnet attributes */
    MagnetData m_mdata;

    /** Propagator algorithm */
    static DipoleAlgorithm<ZLIB::Tps, ZLIB::VTps> s_algorithm;

  };

}

#endif
