// Library       : TEAPOT
// File          : TEAPOT/Integrator/MapDaIntegrator.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_MAP_DA_INTEGRATOR_HH
#define UAL_TEAPOT_MAP_DA_INTEGRATOR_HH

#include "SMF/PacLattice.h"
#include "Optics/PacTMap.h"
#include "Main/Teapot.h"
#include "TEAPOT/Integrator/BasicDaIntegrator.hh"

namespace TEAPOT {

  /** Map evolver. */

  class MapDaIntegrator : public BasicDaIntegrator {

  public:

    /** Constructor */
    MapDaIntegrator();

    /** Copy Constructor */
    MapDaIntegrator(const MapDaIntegrator& mi);

    /** Destructor */
    ~MapDaIntegrator();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Sets lattice elements and generates matrix (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Set the Taylor map */
    void setMap(const PacVTps& vtps);

    /** Propagates a vector of truncated power series */
    void propagate(UAL::Probe& probe);

  protected:

    /* sector length */
    double m_l;

    /** Tylor map */
    PacTMap* m_map;

  private:

    static PacLattice s_lattice;
    static Teapot     s_teapot;

  };

}

#endif
