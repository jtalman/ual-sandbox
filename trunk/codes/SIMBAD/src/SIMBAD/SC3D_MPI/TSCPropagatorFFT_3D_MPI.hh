// Library       : SIMBAD
// File          : SIMBAD/SC3D_MPI/TSCPropagatorFFT_3D_MPI.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_PROPAGATOR_FFT_3D_MPI_HH
#define UAL_SIMBAD_TSC_PROPAGATOR_FFT_3D_MPI_HH

#include "UAL/APF/PropagatorNodePtr.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"
#include "SIMBAD/SC3D_MPI/LoadBalancer.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Propagator */

  class TSCPropagatorFFT_3D_MPI : public TSCPropagatorFFT {

  public:

    /** Constructor */
    TSCPropagatorFFT_3D_MPI();

    /** Copy Constructor */
    TSCPropagatorFFT_3D_MPI(const TSCPropagatorFFT_3D_MPI& c);

    /** Destructor */
    virtual ~TSCPropagatorFFT_3D_MPI();

    /** Returns a deep copy of this object (PropagatorNode method) */
    UAL::PropagatorNode* clone();  

    /** Propagates a bunch (PropagatorNode method) */
    void propagate(UAL::Probe& bunch);
   };


  class TSCPropagatorFFT_3D_MPIRegister 
  {
    public:

    TSCPropagatorFFT_3D_MPIRegister(); 
  };

};

#endif
