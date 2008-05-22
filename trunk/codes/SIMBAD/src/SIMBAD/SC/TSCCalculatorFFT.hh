// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_TSC_CALCULATOR_FFT_HH
#define UAL_SIMBAD_TSC_CALCULATOR_FFT_HH

#include <fftw.h>
#include <complex>
#include <vector>

#include "PAC/Beam/Bunch.hh"

namespace SIMBAD {

  /** FFT-based Transverse Space Charge Calculator */

  class TSCCalculatorFFT {

  public:

    /** Returns a singleton */
    static TSCCalculatorFFT& getInstance();

    /** Destructor */
    virtual ~TSCCalculatorFFT();

    /** Define a grid size */
    void setGridSize(int nxb, int nyb);

    /** Define a max bunch size */
    void setMaxBunchSize(int size);

    /** Define a min bunch size */
    void setMinBunchSize(int size);

    /** Returns a min bunch size */
    int getMinBunchSize() const;  

    /** Define the smoothing parameter */
    void setEps(double eps);

    /** Returns the real number of particles */
    int getBunchSize(const PAC::Bunch& bunch) const;

    /** Returns perveance */
    virtual double getPerveance(const PAC::Bunch& bunch) const;

    /** Calculates force */
    virtual void calculateForce(const PAC::Bunch& bunch);   

    /** Propagates a bunch */
    void propagate(PAC::Bunch& bunch, double l);

    /** Print out the space charge force into the file */
    void showForce(char* file); 

  protected:

   /** Constructor */
   TSCCalculatorFFT();    

  protected:

    void init();
    virtual void defineGridAndRho(const PAC::Bunch& bunch);
    void calculateGreensGrid();
    void fftGreensGrid();
    void fftBunchDensity();
    void calculateForce(double factor); 

  protected:

    static TSCCalculatorFFT* s_theInstance;
    static double rClassical;

  protected:

    double eps;

  protected:

    int nXBins;
    int nYBins;    

    // xGrid used for binning particles, size: nXBins + 1)
    std::vector<double> xGrid;

    // yGrid used for binning particles, size: nYBins + 1)
    std::vector<double> yGrid;

    // Matrix of space charge density, size: [nXBins+1][nYBins+1]
    std::vector<std::vector<double> > rho;

    //  2D array to hold the force in the x direction, size: [nXBins+1][nYBins+1]
    std::vector<std::vector<double> > fscx;

    // 2Darray to hold the force in the y direction, size: [nXBins+1][nYBins+1]
    std::vector<std::vector<double> > fscy;

    std::vector<std::vector<double> > GreensF_re;
    std::vector<std::vector<double> > GreensF_im;

    // FFT structures

    fftwnd_plan planForward;
    fftwnd_plan planBackward;

    // Containers of FFT coefficients 
    // (size: char[nXBins * nYBins * sizeof(FFTW_COMPLEX)])

    FFTW_COMPLEX* in;
    FFTW_COMPLEX* fftRho;
    FFTW_COMPLEX* fftGF;
    FFTW_COMPLEX* fftForce;    

  protected:

    int minBunchSize;
    int maxBunchSize;

    // (size: maxBunchSize)
    std::vector<int> xBin;

    // (size: maxBunchSize)
    std::vector<int> yBin;

    // (size: maxBunchSize_)
    std::vector<double> xFractBinPos;

    // (size: maxBunchSize_)
    std::vector<double> yFractBinPos;

  private:



   };


}

#endif
