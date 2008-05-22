// Library       : SIMBAD
// File          : SIMBAD/SC/TSCPropagatorFFT.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_PROPAGATOR_FFT_MPI_HH
#define UAL_SIMBAD_TSC_PROPAGATOR_FFT_MPI_HH

#include "UAL/APF/PropagatorNodePtr.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Propagator */

  class TSCPropagatorFFT_MPI : public TSCPropagatorFFT {

  public:

    /** Constructor */
    TSCPropagatorFFT_MPI();

    /** Copy Constructor */
    TSCPropagatorFFT_MPI(const TSCPropagatorFFT_MPI& c);

    /** Destructor */
    virtual ~TSCPropagatorFFT_MPI();

    /** Returns a deep copy of this object (PropagatorNode method) */
    UAL::PropagatorNode* clone();  

    /** Propagates a bunch (PropagatorNode method) */
    void propagate(UAL::Probe& bunch);

   };


  class TSCPropagatorFFT_MPIRegister 
  {
    public:

    TSCPropagatorFFT_MPIRegister(); 
  };


}

#endif
