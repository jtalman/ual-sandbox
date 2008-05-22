// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_CALCULATOR_FFT_MPI_HH
#define UAL_SIMBAD_TSC_CALCULATOR_FFT_MPI_HH

#include <fftw.h>
#include <complex>
#include <vector>

#include "PAC/Beam/Bunch.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Calculator */

  class TSCCalculatorFFT_MPI : public  TSCCalculatorFFT {

  public:

    /** Returns a singleton */
    static TSCCalculatorFFT_MPI& getInstance();

    /** Returns perveance */
    virtual double getPerveance(const PAC::Bunch& bunch) const;

   /** Calculates force */
    virtual void calculateForce(const PAC::Bunch& bunch);   

  protected:

   /** Constructor */
   TSCCalculatorFFT_MPI();    

  protected:

    virtual void defineGridAndRho(const PAC::Bunch& bunch);    

  private:

    static TSCCalculatorFFT_MPI* s_theInstanceMPI;

   };


};

#endif
