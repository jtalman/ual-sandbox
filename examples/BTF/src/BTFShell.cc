
#include <iostream>


//UAL Libraries
#include "AIM/BTF/BTFBpmCollector.hh"
#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "PAC/Beam/Bunch.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "Main/Teapot.h"

// ROOT libraries
#include "TNtupleD.h"
#include "TRandom.h"
#include "TH2D.h"

#include "BTFShell.hh"

ClassImp(BTFShell); 

BTFShell::BTFShell()
{
}

BTFShell::~BTFShell()
{ 
} 


UAL::AcceleratorPropagator* BTFShell::getTracker()
{
  return m_ap;
}

double BTFShell::getLength()
{
  Teapot teapot(m_lattice);

  // Survey
  PacSurveyData survey;
  teapot.survey(survey);
  double suml = survey.survey().suml();

  return suml;
}


void BTFShell::generateBunch(PAC::Bunch & bunch, double ctHalfWidth, double deHalfWidth,
			     int iran)
{

  for(int i =0; i < bunch.size(); i++){
    bunch[i].setFlag(0);
    bunch[i].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  ACCSIM::BunchGenerator bunchGenerator;
  int   seed = iran;
  bunchGenerator.addBinomialEllipse1D(bunch, 
				      3,             // m, gauss
				      4,             // ct index
				      ctHalfWidth,   // ct half width
				      5,             // de index
				      deHalfWidth,   // de half width
				      0.0,           // alpha
				      seed);         // seed

}


void BTFShell::getLineDensity(TH2D& signalTH2D)
{  
  const std::list<AIM::BTFSignal>& signals = AIM::BTFBpmCollector::getInstance().getSignals();

  int counter = 0;
  for(std::list<AIM::BTFSignal>::const_iterator it = signals.begin(); it != signals.end(); it++){
    counter++;
    const AIM::BTFSignal& signal = *it;
    for(unsigned int is = 0; is < signal.cts.size(); is++){
      signalTH2D.Fill(counter, signal.cts[is], signal.density[is]);
    } 
  } 
}

void BTFShell::getHDipoleTerm(TH2D& xdtermTH2D)
{  
  const std::list<AIM::BTFSignal>& signals = AIM::BTFBpmCollector::getInstance().getSignals();

  int counter = 0;
  for(std::list<AIM::BTFSignal>::const_iterator it = signals.begin(); it != signals.end(); it++){
    counter++;
    const AIM::BTFSignal& signal = *it;
    for(unsigned int is = 0; is < signal.cts.size(); is++){
      xdtermTH2D.Fill(counter, signal.cts[is], signal.xs[is]);
    } 
  } 
}

void BTFShell::getHSpectrum(TH2D& xspecTH2D, double revFreq, int resonatorNumber)
{  
  const std::list<AIM::BTFSpectrum>& specs = AIM::BTFBpmCollector::getInstance().getHSpectrum();

  double ampl;
  double counter = 0;
  for(std::list<AIM::BTFSpectrum>::const_iterator it = specs.begin(); it != specs.end(); it++){
    counter += 1.0;
    const AIM::BTFSpectrum& spec = *it;
    for(unsigned int is = 0; is < spec.freqs.size(); is++){
      ampl  = spec.values[is].real()*spec.values[is].real();
      ampl += spec.values[is].imag()*spec.values[is].imag();
      xspecTH2D.Fill(counter, (spec.freqs[is]/revFreq - resonatorNumber), ampl);
    } 
  } 
}

/*
double UAL::RootShell::getSlipFactor()
{
  double e = m_ba.getEnergy(), m = m_ba.getMass();
  double v0byc = sqrt(e*e - m*m)/e;

  PAC::Bunch bunch(1);
  bunch.setEnergy(e);

  double de = 4.e-4;
  bunch[0].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, de);
  m_ap->propagate(bunch);

  return -bunch[0].getPosition().getCT()*v0byc/de/m_suml;
}

*/


