// Library       : AIM
// File          : AIM/BTF/BTFSpectrum.hh
// Copyright     : see Copyright file

#ifndef UAL_AIM_BTF_SPECTRUM_HH
#define UAL_AIM_BTF_SPECTRUM_HH

#include <vector>
#include <complex>
  
namespace AIM {

  /** Spectrum of BPM signal*/

  class BTFSpectrum {

  public:

    /** Constructor */
    BTFSpectrum(int nbins = 0);

    /** Set a frequency range */
    void setFreqRange(double freqLo, double freqHi, int nfreqs);

    double ct;

    /** Frequencies */
    std::vector<double> freqs;

    /** Amplitudes */
    std::vector< std::complex<double> > values;

  };

}

#endif
