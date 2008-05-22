// Library       : SIMBAD
// File          : SIMBAD/SC/LSCCalculatorFFT.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_LSC_CALCULATOR_FFT_HH
#define UAL_SIMBAD_LSC_CALCULATOR_FFT_HH

#include <fftw.h>
#include <complex>
#include <vector>

#include "PAC/Beam/Bunch.hh"

namespace SIMBAD {

  /** FFT-based  Longitudinal Space Charge Calculator */

  class LSCCalculatorFFT {

  public:

    /** Returns a singleton */
    static LSCCalculatorFFT& getInstance();

    /** Define a grid size */
    void setGridSize(int nlb);

    /** Define a max bunch size */
    void setMaxBunchSize(int size);

    /** Defines the max size of ct coordinates */
    void setMaxCT(double value) { m_maxCT = value; }

    /** Calculates force */
    void defineLFactors(const PAC::Bunch& bunch);


  private:

   /** Constructor */
   LSCCalculatorFFT();    

  private:

    static LSCCalculatorFFT* s_theInstance;

  private:

  // RF frequency
  double m_maxCT;

  private:

    int m_maxBunchSize;

    // (size: m_maxBunchSize)
    std::vector<int> m_bins;

    // (size: m_maxBunchSize)
    std::vector<double> m_fractBins;

  private:

    int m_nBins;

    // Vector of longitudinal grid values (size: m_nBins + 1)
    std::vector<double> m_grid;

    // Vector of line space charge density (size: m_nBins + 1)
    std::vector<double> m_rho;
 
   };


}

#endif
