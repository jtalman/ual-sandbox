
#include <rfftw.h>
#include "AIM/BPM/MonitorCollector.hh"

AIM::MonitorCollector* AIM::MonitorCollector::s_theInstance = 0;

AIM::MonitorCollector::MonitorCollector()
{
}

AIM::MonitorCollector& AIM::MonitorCollector::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new AIM::MonitorCollector();
  }
  return *s_theInstance;
}

void AIM::MonitorCollector::registerBPM(AIM::Monitor* bpm)
{
  std::map<int, AIM::Monitor*>::iterator it = m_bpms.find(bpm->getIndex());
  if(it != m_bpms.end()){
    std::cout << "Monitor " << bpm->getName() 
	      << " has been already registered " << std::endl;
    return;
  }

  m_bpms[bpm->getIndex()] = bpm;

}

std::map<int, AIM::Monitor*>&  AIM::MonitorCollector::getAllData()
{
  return m_bpms;
}


void AIM::MonitorCollector::clear()
{
  std::map<int, AIM::Monitor*>::iterator ibpm;
  for(ibpm = m_bpms.begin(); ibpm != m_bpms.end(); ibpm++){
    ibpm->second->clear();
  }
}

void AIM::MonitorCollector::write(const char* fileName)
{


  m_file.open(fileName);
  if(!m_file) {
    std::cerr << "Cannot open MonitorCollector output file " << fileName << std::endl;
    return;
  }

  if(m_bpms.begin() == m_bpms.end()) {
    m_file.close();
    return;
  }

  m_file << "lattice: " << m_bpms.begin()->second->getLatticeName() << std::endl;
  m_file << "nTurns: "  << m_bpms.begin()->second->getTurns() << std::endl;
 
  m_file << std::endl;

  int counter = 0;
  std::map<int, AIM::Monitor*>::iterator ibpm;
  for(ibpm = m_bpms.begin(); ibpm != m_bpms.end(); ibpm++){

    m_file <<  "BPM index : " << counter++ << std::endl;
    ibpm->second->write(m_file); 
    m_file << "\n\n";

  }

  m_file.close();

}

void AIM::MonitorCollector::fft(int bpmIndex, 
				std::vector<double>& freq,
				std::vector<double>& hspec, 
				std::vector<double>& vspec)
{
  std::map<int, AIM::Monitor*>::iterator ibpm = m_bpms.find(bpmIndex);
  if(ibpm == m_bpms.end()) return;

  fft(ibpm->second->getData(), freq, hspec, vspec);
}


void AIM::MonitorCollector::fft(std::list<PAC::Position>& tbt, 
				std::vector<double>& freq,
				std::vector<double>& hspec, 
				std::vector<double>& vspec)
{
  int turns = tbt.size();

  fftw_real* xs    = new fftw_real[turns];
  fftw_real* ys    = new fftw_real[turns];
  fftw_real* xffts = new fftw_real[turns];
  fftw_real* yffts = new fftw_real[turns];

  fftw_real* xpower_spectrum = new fftw_real[turns];
  fftw_real* ypower_spectrum = new fftw_real[turns];

  int counter = 0;
  std::list<PAC::Position>::iterator it;

  for(it = tbt.begin(); it != tbt.end(); it++){
    xs[counter++] = it->getX();
  }

  counter = 0;
  for(it = tbt.begin(); it != tbt.end(); it++){
    ys[counter++] = it->getY();
  }

  rfftw_plan fftplan = rfftw_create_plan(turns, 
					 FFTW_REAL_TO_COMPLEX, 
					 FFTW_ESTIMATE);

  rfftw_one(fftplan, xs, xffts);
  rfftw_one(fftplan, ys, yffts);

  xpower_spectrum[0] = xffts[0]*xffts[0];  /* DC component */
  ypower_spectrum[0] = yffts[0]*yffts[0];  /* DC component */

  for (int k = 1; k < (turns+1)/2; ++k){  /* (k < N/2 rounded up) */
    xpower_spectrum[k] = xffts[k]*xffts[k] + xffts[turns-k]*xffts[turns-k];
    ypower_spectrum[k] = yffts[k]*yffts[k] + yffts[turns-k]*yffts[turns-k];
  }
  
  /* Nyquist freq. */
  if (turns % 2 == 0) { /* N is even */
    xpower_spectrum[turns/2] = xffts[turns/2]*xffts[turns/2];
    ypower_spectrum[turns/2] = yffts[turns/2]*yffts[turns/2];
  }

  freq.resize((turns+1)/2);
  hspec.resize((turns+1)/2);
  vspec.resize((turns+1)/2);

  for(int it = 0; it < (turns+1)/2; it++){

    // std::cout << it << " " 
    // << xpower_spectrum[it] << " " 
    // << ypower_spectrum[it] << std::endl;
    freq[it]  = (1.0*it)/turns;
    hspec[it] = xpower_spectrum[it];
    vspec[it] = ypower_spectrum[it];
  }

  rfftw_destroy_plan(fftplan);

  delete [] xs;
  delete [] ys;
  delete [] xffts;
  delete [] yffts;

  delete [] xpower_spectrum;
  delete [] ypower_spectrum; 

}




