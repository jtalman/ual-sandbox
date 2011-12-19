// Library       : TEAPOT
// File          : TEAPOT/Integrator/DriftDaIntegrator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DRIFT_DA_INTEGRATOR_HH
#define UAL_TEAPOT_DRIFT_DA_INTEGRATOR_HH

#include "TEAPOT/Integrator/BasicDaIntegrator.hh"
#include "TEAPOT/Integrator/DriftAlgorithm.hh"

namespace TEAPOT {

  /** Drift DA integrator. */

  class DriftDaIntegrator : public BasicDaIntegrator {

  public:

    /** Constructor */
    DriftDaIntegrator();

    /** Copy constructor */
    DriftDaIntegrator(const DriftDaIntegrator& dt);

    /** Destructor */
    ~DriftDaIntegrator();

    const char* getType();

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


    /** Element length */
    double m_l;

    /** Propagator algorithm */
    static DriftAlgorithm<ZLIB::Tps, ZLIB::VTps> s_algorithm;

  private:

    void initialize();
    void copy(const DriftDaIntegrator& di);

  };

}

#endif
