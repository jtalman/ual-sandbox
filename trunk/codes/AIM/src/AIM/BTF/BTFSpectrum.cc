// Library       : AIM
// File          : AIM/BTF/BTFSpectrum.cc
// Copyright     : see Copyright file

#include "AIM/BTF/BTFSpectrum.hh"

AIM::BTFSpectrum::BTFSpectrum(int nbins)
  : freqs(nbins), values(nbins)
{
}

void AIM::BTFSpectrum::setFreqRange(double freqLo, double freqHi, int nfreqs)
{
  freqs.resize(nfreqs);
  values.resize(nfreqs);

  std::complex<double> czero(0.0, 0.0);
  for(int i=0; i < nfreqs; i++){
    freqs[i]  = freqLo + i*(freqHi - freqLo)/(nfreqs - 1);;
    values[i] = czero;
  }  

  ct = 0.0;
}
