
#include "UAL/SXF/Parser.hh"
#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "Optics/PacChromData.h"

#include "AIM/BPM/MonitorCollector.hh"
#include "AIM/BPM/PoincareMonitorCollector.hh"

#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/SeparatrixCalculator.hh"
#include "UAL/QT/Player/TurnCounter.hh"

UAL::QT::PlayerShell::PlayerShell()
{
  m_V      = 0.0;
  m_harmon = 1.0;
  m_lag    = 0.0;
}

/*

void UAL::QT::PlayerShell::getTwiss(PacTwissData& twiss)
{
  twiss = UAL::OpticsCalculator::getInstance().m_chrom->twiss();
}

*/

UAL::QT::PlayerShell::~PlayerShell()
{
}

bool UAL::QT::PlayerShell::setBeamAttributes(const UAL::Arguments& arguments)
{
  bool res = UAL::Shell::setBeamAttributes(arguments);
  if(res == false) return res;

  UAL::SeparatrixCalculator::getInstance().setBeamAttributes(m_ba);

  return res;
}


bool UAL::QT::PlayerShell::setRF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  it = args.find("V");
  if(it != args.end()) m_V = it->second->getNumber();

  it = args.find("harmon");
  if(it != args.end()) m_harmon = it->second->getNumber();

  it = args.find("lag");
  if(it != args.end()) m_lag = it->second->getNumber();

  // m_separatrix.setRFCavity(m_V, m_harmon, m_lag);

  return true;
}


bool UAL::QT::PlayerShell::initRun(const UAL::Arguments& arguments)
{

  std::cout << "UAL::QT::PlayerShell::initRun(): " << std::endl;

  // std::cout << "+update lattice  " << std::endl;

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  bool flag = optics.calculate();
  if(!flag)  return flag;

  double charge  = m_ba.getCharge();
  double m       = m_ba.getMass();
  double e       = m_ba.getEnergy();

  double p2      = e*e - m*m;
  double gamma   = e/m;
  double v0byc2  = p2/(e*e);
  double v0byc   = sqrt(v0byc2);

  std::cout << "UAL::QT::PlayerShell::initRun(): update separatrix  " << std::endl;

  double eta0   = optics.alpha0 - 1.0/gamma/gamma;

  // std::cout << "before " << ", gamma = " << gamma 
  //	    << ", gt = " << 1./sqrt(optics.alpha0) 
  //	    << ", m_lag = " << m_lag << std::endl;

  if(eta0 < 0 && m_lag > 0.25) m_lag = 0.5 - m_lag;
  if(eta0 > 0 && m_lag < 0.25) m_lag = 0.5 - m_lag;  

  // std::cout << "after " << ", gamma = " << gamma 
  //	    << ", gt = " << 1./sqrt(optics.alpha0) 
  //	    << ", m_lag = " << m_lag << std::endl;

  UAL::SeparatrixCalculator& separatrix = 
    UAL::SeparatrixCalculator::getInstance();

  separatrix.setRFCavity(m_V, m_harmon, m_lag);
  separatrix.setLattice(optics.suml, optics.alpha0);
  separatrix.calculate();

  // std::cout << "+update bunch  " << std::endl;

  if(m_bunchGenerator.ctHalfWidth < 0 || m_bunchGenerator.deHalfWidth < 0) {
    m_bunchGenerator.ctHalfWidth = optics.suml/m_harmon/2./2.;
    m_bunchGenerator.deHalfWidth = separatrix.getDeMax()/2.;
  }

  updateBunch();
  m_bunch.getBeamAttributes().setRevfreq(UAL::clight*v0byc/optics.suml);

  std::cout << "UAL::QT::PlayerShell::initRun(): update propagator  " << std::endl;

  // m_sectorTracker.setOptics(optics);

  AIM::MonitorCollector::getInstance().getAllData().clear();
  AIM::PoincareMonitorCollector::getInstance().getAllData().clear();
  rebuildPropagator();
  m_rfTracker.setRF(m_V, m_harmon, m_lag);

  return true;

}


bool UAL::QT::PlayerShell::run(int turn)
{

  TurnCounter::getInstance()->setTurn(turn);

  PAC::BeamAttributes& ba = m_bunch.getBeamAttributes();

  double m       = ba.getMass();
  double e       = ba.getEnergy();
  double gamma   = e/m;

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
  double eta0   = optics.alpha0 - 1.0/gamma/gamma;

  // if((turn/100)*100 == turn) {
  //  std::cout << "eta = " << eta0 << ", m_lag = " << m_lag << std::endl;
  // }


  if(eta0 < 0 && m_lag > 0.25) {
    changeLag(optics);
  }
  if(eta0 > 0 && m_lag < 0.25) {
    changeLag(optics);
  }

  m_ap->propagate(m_bunch);
  // m_sectorTracker.propagate(m_bunch);
  m_rfTracker.propagate(m_bunch);

  return true;

}

void UAL::QT::PlayerShell::changeLag(UAL::OpticsCalculator& optics)
{

  UAL::SeparatrixCalculator& separatrix = UAL::SeparatrixCalculator::getInstance();

  m_lag = 0.5 - m_lag;

  std::cout << "RF phase has been changed" 
	    << ", a new  phase = " <<  m_lag*360.0
	    << "degrees" << std::endl;

  m_rfTracker.setRF(m_V, m_harmon, m_lag);

  separatrix.setRFCavity(m_V, m_harmon, m_lag);
  separatrix.setLattice(optics.suml, optics.alpha0);
  separatrix.calculate();
}

void UAL::QT::PlayerShell::setLongEmittance(double ctMax, double deMax)
{
  m_bunchGenerator.ctHalfWidth = ctMax; 
  m_bunchGenerator.deHalfWidth = deMax;
}





