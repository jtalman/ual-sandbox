// Library       : AIM
// File          : AIM/BTF/BTFBpm.cc
// Copyright     : see Copyright file
// Author        : P.Cameron and M.Blaskiewicz
// C++ version   : N.Malitsky 

#include "AIM/BTF/BTFBpm.hh"
#include "AIM/BTF/BTFBpmCollector.hh"

AIM::BTFBpm::BTFBpm()
{
  init();
}

AIM::BTFBpm::BTFBpm(const AIM::BTFBpm& bpm)
{
  copy(bpm);
}

void AIM::BTFBpm::init()
{
  m_ctBin   = 0;  
  m_tau     = 1.0e+21;

  m_hFreqLo = 0;
  m_hFreqHi = 0;
  m_hNFreqs = 0;

  m_vFreqLo = 0;
  m_vFreqHi = 0;
  m_vNFreqs = 0;  
}

void AIM::BTFBpm::copy(const AIM::BTFBpm& bpm)
{
  m_ctBin   = bpm.m_ctBin;  
  m_tau     = bpm.m_tau;

  m_hFreqLo = bpm.m_hFreqLo;
  m_hFreqHi = bpm.m_hFreqHi;
  m_hNFreqs = bpm.m_hNFreqs;

  m_vFreqLo = bpm.m_vFreqLo;
  m_vFreqHi = bpm.m_vFreqHi;
  m_vNFreqs = bpm.m_vNFreqs;  
}

void AIM::BTFBpm::setCtBin(double ctBin)
{
  m_ctBin = ctBin;
}

void AIM::BTFBpm::setTau(double tau)
{
  m_tau = tau;
}

void AIM::BTFBpm::setHFreqRange(double freqLo, double freqHi, int nfreqs)
{
  m_hFreqLo = freqLo;
  m_hFreqHi = freqHi;
  m_hNFreqs = nfreqs;  
}

void AIM::BTFBpm::setVFreqRange(double freqLo, double freqHi, int nfreqs)
{
  m_vFreqLo = freqLo;
  m_vFreqHi = freqHi;
  m_vNFreqs = nfreqs;  
}

UAL::PropagatorNode* AIM::BTFBpm::clone()
{
  return new AIM::BTFBpm(*this);
}

void AIM::BTFBpm::propagate(UAL::Probe& probe)
{
  
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe); 
  double revFreq = bunch.getBeamAttributes().getRevfreq();

  AIM::BTFBpmCollector& bpmCollector = AIM::BTFBpmCollector::getInstance();

  // Find signal (line density + dipole terms)

  AIM::BTFSignal signal;
  bunch2signal(bunch, signal);
  bpmCollector.addSignal(signal);

  // Find horizontal spectrum

  const std::list<AIM::BTFSpectrum>& hSpecs = bpmCollector.getHSpectrum();

  AIM::BTFSpectrum hSpecOut;
  hSpecOut.setFreqRange(m_hFreqLo, m_hFreqHi, m_hNFreqs);

  if(hSpecs.empty()){
     AIM::BTFSpectrum hSpecIn;
     hSpecIn.setFreqRange(m_hFreqLo, m_hFreqHi, m_hNFreqs);
     bpmCollector.addHSpectrum(hSpecIn);
  }

  signal2spectrum(signal, 1./revFreq, hSpecs.back(), hSpecOut);
  bpmCollector.addHSpectrum(hSpecOut);

  // Find vertical  spectrum

  const std::list<AIM::BTFSpectrum>& vSpecs = bpmCollector.getVSpectrum();

  AIM::BTFSpectrum vSpecOut;
  vSpecOut.setFreqRange(m_vFreqLo, m_vFreqHi, m_vNFreqs);

  if(vSpecs.empty()){
     AIM::BTFSpectrum vSpecIn;
     vSpecIn.setFreqRange(m_vFreqLo, m_vFreqHi, m_vNFreqs);
     bpmCollector.addVSpectrum(vSpecIn);
  }

  signal2spectrum(signal, 1./revFreq, vSpecs.back(), vSpecOut);
  bpmCollector.addVSpectrum(vSpecOut);  

}

void AIM::BTFBpm::bunch2signal(PAC::Bunch& bunch, AIM::BTFSignal& signal)
{

  if(m_ctBin == 0.0){
    signal.resize(0);
    return; 
  }

  // Find the longitudinal size of the bunch

  double ct;
  double ctLo =  1.0e+21;
  double ctHi = -1.0e+21;
  for(int ip = 0; ip < bunch.size(); ip++){
    if(!bunch[ip].isLost()) {
      ct = bunch[ip].getPosition().getCT();
      if( ct > ctHi ) ctHi = ct;   
      if( ct < ctLo ) ctLo = ct;
    }
  }

  int nbins = int((ctHi - ctLo)/m_ctBin);
  if((ctHi - ctLo)/m_ctBin > float(nbins)) nbins++;
  nbins++;

  signal.resize(nbins);

  // initialize signal containers

  for(int is = 0; is < nbins; is++){
    signal.cts[is]     = ctLo + is*m_ctBin;
    signal.density[is] = 0.0;
    signal.xs[is]      = 0.0;
    signal.ys[is]      = 0.0;
  }

  // calcualtes line density and dipole driving terms 
  // using linear interpolation

  int nlo, nhi;
  double ctk, tk, flo, fhi, xk, yk;
  
  for(int k = 0; k < bunch.size(); k++){

    if(bunch[k].isLost()) continue;

    PAC::Position& pos = bunch[k].getPosition();

    ctk = pos.getCT();
    tk  = (ctk - ctLo)/m_ctBin;

    nlo = int(tk);
    nhi = nlo+1;

    if(nhi >= nbins) nhi = nbins-1;

    flo = nhi-tk;
    fhi = 1-flo;
   
    signal.density[nlo] += flo;
    signal.density[nhi] += fhi;

    xk = pos.getX();
    signal.xs[nlo] += xk*flo;
    signal.xs[nhi] += xk*fhi;

    yk = pos.getY();
    signal.ys[nlo] += yk*flo;
    signal.ys[nhi] += yk*fhi;
    
  }
}

void AIM::BTFBpm::signal2spectrum(AIM::BTFSignal& signal, double T, 
				  const AIM::BTFSpectrum& specin, AIM::BTFSpectrum& specout )
{
  
  double dt = 0;
  double dT = T + specin.ct/UAL::clight - signal.cts[signal.cts.size() - 1]/UAL::clight;

  std::complex<double> zsk;
  for(unsigned int kspec = 0; kspec < specin.freqs.size(); kspec++){

    double omegak = 2.*UAL::pi*specin.freqs[kspec];

    std::complex<double> zphi0(0.0, omegak*dT);
    zphi0 = exp(zphi0)*exp(-dT/m_tau);
    zsk   = specin.values[kspec]*zphi0;

    dt = m_ctBin/UAL::clight;
    std::complex<double> zphi(0.0, omegak*dt);
    zphi = exp(zphi)*exp(- m_ctBin/m_tau);

    zsk += signal.xs[signal.cts.size()-1];
    for(int kpart = signal.cts.size()-2; kpart >= 0; kpart--){ 
      zsk   = zphi*zsk; 
      zsk += signal.xs[kpart];
    }
    specout.values[kspec] = zsk; 
  }
  specout.ct = signal.cts[0];
}


