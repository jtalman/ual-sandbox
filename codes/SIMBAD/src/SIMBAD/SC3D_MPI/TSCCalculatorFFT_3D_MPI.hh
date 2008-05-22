// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT_3D_MPI.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_CALCULATOR_FFT_3D_MPI_HH
#define UAL_SIMBAD_TSC_CALCULATOR_FFT_3D_MPI_HH

#include <fftw.h>
#include <complex>
#include <vector>

#include "PAC/Beam/Bunch.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Calculator */

  class TSCCalculatorFFT_3D_MPI : public  TSCCalculatorFFT {

  public:

    /** Returns a singleton */
    static TSCCalculatorFFT_3D_MPI& getInstance();

    /** Returns perveance */
    virtual double getPerveance(PAC::Bunch& bunch,
				vector<int>& subBunchIndices) const;

    /** Calculates force */
    virtual void calculateForce(PAC::Bunch& bunch,
				vector<int>& subBunchIndices);   

    /** Propogates a sub bunch */
    void propagate(PAC::Bunch& bunch,
		   vector<int>& subBunchIndices,
		   double length);

  protected:

   /** Constructor */
   TSCCalculatorFFT_3D_MPI();

  protected:

    virtual void defineGridAndRho(const PAC::Bunch& bunch,
				  vector<int>& subBunchIndices);

  private:
    static TSCCalculatorFFT_3D_MPI* s_theInstanceMPI;

   };


};

#endif
