// Library       : SIMBAD
// File          : SIMBAD/SC/TSCPropagatorFFT.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_PROPAGATOR_FFT_HH
#define UAL_SIMBAD_TSC_PROPAGATOR_FFT_HH

#include "UAL/APF/PropagatorNodePtr.hh"
#include "SIMBAD/SC/TSCPropagator.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Propagator */

  class TSCPropagatorFFT : public SIMBAD::TSCPropagator {

  public:

    /** Constructor */
    TSCPropagatorFFT();

    /** Copy Constructor */
    TSCPropagatorFFT(const TSCPropagatorFFT& c);

    /** Destructor */
    virtual ~TSCPropagatorFFT();


    /** Returns a deep copy of this object (PropagatorNode method) */
    UAL::PropagatorNode* clone();  

    /** Defines the lattice elemements (PropagatorNode method)
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch (PropagatorNode method) */
    void propagate(UAL::Probe& bunch);

    /** Sets length */
    void setLength(double l);

    /** Returns length */
    double getLength() const;


  protected:

    /** Conventional tracker */
    UAL::PropagatorNodePtr m_tracker;

    /** length */
    double m_lkick;

   };


  class TSCPropagatorFFTRegister 
  {
    public:

    TSCPropagatorFFTRegister(); 
  };


}

#endif
