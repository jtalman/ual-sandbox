
#include <rfftw.h>
#include "AIM/BPM/PoincareMonitorCollector.hh"

AIM::PoincareMonitorCollector* AIM::PoincareMonitorCollector::s_theInstance = 0;

AIM::PoincareMonitorCollector::PoincareMonitorCollector()
{
}

AIM::PoincareMonitorCollector& AIM::PoincareMonitorCollector::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new AIM::PoincareMonitorCollector();
  }
  return *s_theInstance;
}

void AIM::PoincareMonitorCollector::registerBPM(AIM::PoincareMonitor* bpm)
{
  std::map<int, AIM::PoincareMonitor*>::iterator it = m_bpms.find(bpm->getIndex());
  if(it != m_bpms.end()){
    std::cout << "BPM " << bpm->getName() 
	      << " has been already registered " << std::endl;
    return;
  }

  m_bpms[bpm->getIndex()] = bpm;

}

std::map<int, AIM::PoincareMonitor*>&  AIM::PoincareMonitorCollector::getAllData()
{
  return m_bpms;
}


void AIM::PoincareMonitorCollector::clear()
{
  std::map<int, AIM::PoincareMonitor*>::iterator ibpm;
  for(ibpm = m_bpms.begin(); ibpm != m_bpms.end(); ibpm++){
    ibpm->second->clear();
  }
}


void AIM::PoincareMonitorCollector::fft(int bpmIndex, 
					std::vector<double>& xFreq,
					std::vector<double>& yFreq)
{
  std::map<int, AIM::PoincareMonitor*>::iterator ibpm = m_bpms.find(bpmIndex);
  if(ibpm == m_bpms.end()) return;

  fft(ibpm->second->getData(), xFreq, yFreq);
}


void AIM::PoincareMonitorCollector::fft(std::list<PAC::Bunch>& tbt, 
					std::vector<double>& xFreq,
					std::vector<double>& yFreq)
{
  int nturns = tbt.size();
  if(nturns < 1) return;

  std::cout << "AIM::PoincareMonitorCollector: nturns = " << nturns << std::endl;

  int np = tbt.back().size();

  xFreq.resize(np);
  yFreq.resize(np);

  // int nturns = 128; // 1024;

  fftw_real** xs    = new fftw_real*[np];
  fftw_real** ys    = new fftw_real*[np];
  fftw_real** xffts = new fftw_real*[np];
  fftw_real** yffts = new fftw_real*[np];

  fftw_real** xpower_spectrum = new fftw_real*[np];
  fftw_real** ypower_spectrum = new fftw_real*[np];

  int ip;
  for(ip = 0; ip < np; ip++){
    xs[ip]    = new fftw_real[nturns];
    ys[ip]    = new fftw_real[nturns];
    xffts[ip] = new fftw_real[nturns];
    yffts[ip] = new fftw_real[nturns];

    xpower_spectrum[ip] = new fftw_real[nturns];
    ypower_spectrum[ip] = new fftw_real[nturns];
  }

  int it = 0;
  for(std::list<PAC::Bunch>::iterator il = tbt.begin(); il != tbt.end(); il++){
    PAC::Bunch& bunch = *il;
    for(ip =0; ip < np; ip++){
      if(bunch[ip].isLost()) continue;
      PAC::Position& pos = bunch[ip].getPosition();
      xs[ip][it] = pos.getX();
      ys[ip][it] = pos.getY();
    }
    it++;
  }

  rfftw_plan fftplan = rfftw_create_plan(nturns, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

  // int pCounter = 0;
  for(ip = 0; ip < np; ip++){

    if(tbt.back()[ip].isLost()) continue;

    rfftw_one(fftplan, xs[ip], xffts[ip]);
    rfftw_one(fftplan, ys[ip], yffts[ip]);

    xpower_spectrum[ip][0] = xffts[ip][0]*xffts[ip][0];  /* DC component */
    ypower_spectrum[ip][0] = yffts[ip][0]*yffts[ip][0];  /* DC component */

    for (int k = 1; k < (nturns+1)/2; ++k){  /* (k < N/2 rounded up) */
      xpower_spectrum[ip][k] = xffts[ip][k]*xffts[ip][k] + xffts[ip][nturns-k]*xffts[ip][nturns-k];
      ypower_spectrum[ip][k] = yffts[ip][k]*yffts[ip][k] + yffts[ip][nturns-k]*yffts[ip][nturns-k];
    }
    if (nturns % 2 == 0) { /* N is even */
       xpower_spectrum[ip][nturns/2] = xffts[ip][nturns/2]*xffts[ip][nturns/2];  /* Nyquist freq. */
       ypower_spectrum[ip][nturns/2] = yffts[ip][nturns/2]*yffts[ip][nturns/2];  /* Nyquist freq. */
    }
  }

  // double tunex = qx - iqx;
  // double tuney = qy - iqy; 

  double xTune, xMinTune, xMaxTune; 
  double yTune, yMinTune, yMaxTune;

  xMinTune = yMinTune = +1.0;
  xMaxTune = yMaxTune = -1.0;

  for(ip = 0; ip < np; ip++){

    if(tbt.back()[ip].isLost()) {
      // std::cout << ip << " particle is lost: " << exi[ip] << " " << eyi[ip] << std::endl;
      std::cout << ip << " particle is lost: " << std::endl;
      continue;
    }

    double ixfftmax   = 0;
    double iyfftmax   = 0;
    double xfftmax    = 0.0;
    double yfftmax    = 0.0;

    for(it = 1; it < (nturns+1)/2; it++){

      if(xfftmax < xpower_spectrum[ip][it]) {
	xfftmax  = xpower_spectrum[ip][it];
	ixfftmax = it;
      }
      if(yfftmax < ypower_spectrum[ip][it]) {
	yfftmax  = ypower_spectrum[ip][it];
	iyfftmax = it;
      }
    }

    xTune = ixfftmax/nturns; //  - tunex;
    yTune = iyfftmax/nturns; //  - tuney;

    std::cout << ip << " " 
	      << ", ix = " << ixfftmax << ", tunex= " << xTune 
	      << ", iy = " << iyfftmax << ", tuney= " << yTune  << std::endl;

    xFreq[ip] = xTune;
    yFreq[ip] = yTune;

    if(xMaxTune < xTune) xMaxTune = xTune;
    if(xMinTune > xTune) xMinTune = xTune;
    if(yMaxTune < yTune) yMaxTune = yTune;
    if(yMinTune > yTune) yMinTune = yTune;
  }

  // std::cout << "xMaxTune = " << xMaxTune << ", xMinTune = " << xMinTune << std::endl;
  // std::cout << "yMaxTune = " << yMaxTune << ", yMinTune = " << yMinTune << std::endl;

  rfftw_destroy_plan(fftplan);

  for(ip = 0; ip < np; ip++){
    delete [] xs[ip];
    delete [] ys[ip];
    delete [] xffts[ip];
    delete [] yffts[ip];
    delete [] xpower_spectrum[ip];
    delete [] ypower_spectrum[ip];
  }

  delete [] xs;
  delete [] ys;
  delete [] xffts;
  delete [] yffts;
  delete [] xpower_spectrum;
  delete [] ypower_spectrum;

}


