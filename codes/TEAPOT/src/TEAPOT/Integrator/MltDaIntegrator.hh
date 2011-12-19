// Library       : TEAPOT
// File          : TEAPOT/Integrator/MltDaIntegrator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_MLT_DA_INTEGRATOR_HH
#define UAL_TEAPOT_MLT_DA_INTEGRATOR_HH

#include "SMF/PacElemMultipole.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/MagnetAlgorithm.hh"
#include "TEAPOT/Integrator/BasicDaIntegrator.hh"

namespace TEAPOT {

  /** Multipole DA integrator. */

  class MltDaIntegrator : public BasicDaIntegrator {

  public:

    /** Constructor */
    MltDaIntegrator();

    /** Copy constructor */
    MltDaIntegrator(const MltDaIntegrator& mi);

    /** Destructor */
    ~MltDaIntegrator();

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

    /** Complexity number */
    double m_ir;

    /** Magnet attributes */
    MagnetData m_mdata;

    /** Propagator algorithm */
    static MagnetAlgorithm<ZLIB::Tps, ZLIB::VTps> s_algorithm;

  private:

    void initialize();
    void copy(const MltDaIntegrator& mi);

  };

}

#endif
