#include <cmath>

#include "UAL/Common/Def.hh"
#include "Optics/PacChromData.h"
#include "PAC/Beam/Bunch.hh"

#include "UAL/QT/Player/SeparatrixCalculator.hh"

UAL::SeparatrixCalculator* UAL::SeparatrixCalculator::s_theInstance = 0;

UAL::SeparatrixCalculator& UAL::SeparatrixCalculator::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new UAL::SeparatrixCalculator();
  }
  return *s_theInstance;
}

UAL::SeparatrixCalculator::SeparatrixCalculator()
  : m_phases(40), m_des(40)
{
  m_suml   = 0.0;
  m_alpha0 = 1.0;
  m_rfTracker.setRF(0.0, 1.0, 0.0);
}

void UAL::SeparatrixCalculator::setBeamAttributes(const PAC::BeamAttributes& ba)
{
  m_ba = ba;
}

void UAL::SeparatrixCalculator::setLattice(double suml, double alpha0)
{
  m_suml   = suml;
  m_alpha0 = alpha0;
}

void UAL::SeparatrixCalculator::setRFCavity(double V, double harmon, double lag)
{
  m_rfTracker.setRF(V, harmon, lag);
}

bool UAL::SeparatrixCalculator::calculate()
{

  if(isNearTransition()) {

    for(unsigned int i = 0; i < m_phases.size(); i++){
      m_phases[i] = 0;
      m_des[i] = 0;
    }

    return true;
  }
  
  // calculateBucket();

  double e      = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (e*e - m*m)/e/e;
  double v0byc  = sqrt(v0byc2);
  double p2     = e*e - m*m;
  double gamma  = e/m;
  double gamma2 = gamma*gamma;

  double eta0   = m_alpha0 - 1.0/gamma2;

  double harmon = m_rfTracker.getHarmon();
  double lag    = m_rfTracker.getLag();

  double phaseS = lag*(2.0*UAL::pi);
  // double ct2phi = (2.0*UAL::pi*harmon)*(v0byc/m_suml);

  double Hsep; 

  if(eta0 < 0) {
    if(lag > -0.25 && lag < 0.25){
      Hsep   = getHsep(UAL::pi - phaseS); 
    } else {
      Hsep   = getHsep(phaseS); 
    }
  } else {
    if(lag > -0.25 && lag < 0.25){
      Hsep   = getHsep(phaseS); 
    } else {
      Hsep   = getHsep(UAL::pi - phaseS); 
    }    
  }

  // std::cout << "phaseS = " << phaseS << std::endl;
 
  double phase, de, de2;

  int size = m_phases.size();

  for(int i = 0; i < size; i++){

    phase   = (phaseS - UAL::pi) + i*(2.0*UAL::pi)/(size - 1);
    de2  = Hsep - getV(phase);
    // std::cout << i << " " << phase  << ", Hsep = " << Hsep 
    // << ", V = " << getV(phase) << std::endl;
  
    if(de2 < 0.0) de = 0.0;
    else de = sqrt(de2);

    m_phases[i] = phase; // -phase/ct2phi;
    m_des[i] = de;
    // std::cout << i << " cts=" << m_phases[i]  
    // << ", de = " << m_des[i] << std::endl;

  }
  return true;
}

double UAL::SeparatrixCalculator::getHsep(double phaseU)
{
  double Hsep = 0.0;
  Hsep += getHsep(phaseU, 
		  m_rfTracker.getV(), 
		  m_rfTracker.getHarmon(), 
		  m_rfTracker.getLag());
  return Hsep;
}

double UAL::SeparatrixCalculator::getHsep(double phaseU, 
					  double V, 
					  double harmon, 
					  double lag)
{

  if(harmon == 0) return 0;

  double charge = m_ba.getCharge();
  double e      = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (e*e - m*m)/e/e;
  double p2     = e*e - m*m;
  double gamma  = e/m;
  double gamma2 = gamma*gamma;

  double eta0   = m_alpha0 - 1.0/gamma2;

  double phaseS = lag*(2.*UAL::pi);

  double Hsep   = (cos(phaseU) - cos(phaseS) + (phaseU - phaseS)*sin(phaseS));
  double coeff  = (charge*V*2.0*e*v0byc2)/(2.0*UAL::pi*harmon*eta0*p2);

  return Hsep*coeff;
}

double UAL::SeparatrixCalculator::getV(double phase)
{
  double result = 0.0;
  result += getV(phase, 
		 m_rfTracker.getV(), 
		 m_rfTracker.getHarmon(), 
		 m_rfTracker.getLag());
  return result;
  
}

double UAL::SeparatrixCalculator::getV(double phase,
				       double V, 
				       double harmon, 
				       double lag)
{

  if(harmon == 0) return 0;

  double charge = m_ba.getCharge();
  double e      = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (e*e - m*m)/e/e;
  double p2     = e*e - m*m;
  double gamma  = e/m;
  double gamma2 = gamma*gamma;

  double eta0   = m_alpha0 - 1.0/gamma2;

  double phaseS = lag*(2.*UAL::pi);

  // double ct2phi = (2.0*UAL::pi*harmon)*(v0byc/m_suml);
  // double phase  = -ct*ct2phi + phaseS;

  double result = (cos(phase) - cos(phaseS) + (phase - phaseS)*sin(phaseS));
  double coeff  = (charge*V*2.0*e*v0byc2)/(2.0*UAL::pi*harmon*eta0*p2);

  return result*coeff;
}

bool UAL::SeparatrixCalculator::isNearTransition()
{
  double charge = m_ba.getCharge();
  double energy = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (energy*energy - m*m)/energy/energy;
  double v0byc  = sqrt(v0byc2);
  double gamma  = energy/m;

  double V      = m_rfTracker.getV();
  double harmon = m_rfTracker.getHarmon();
  double lag    = m_rfTracker.getLag();

  double gt     = 1./sqrt(m_alpha0);

  double phaseS = lag*(2.*UAL::pi);
  
  double revFreq= UAL::clight*v0byc/m_suml;
  double omega  = 2*UAL::pi*revFreq;
  double omega2 = omega*omega;

  double dgamma = abs(charge*V*sin(phaseS)/m)*revFreq;

  if(lag == 0.5 || lag == -0.5) {
    if(gamma == gt) return true;
    else return false;
  }

  // std::cout << "dgamma = " << dgamma << std::endl;

  double Tc = UAL::pi*energy*v0byc2;
  Tc       /= charge*V*cos(phaseS)*dgamma*harmon*omega2;
  Tc        = abs(Tc);
  Tc        = gt*pow(Tc, 1./3.);

  if(abs(gamma - gt)/dgamma < Tc) {
    /* 
    std::cout << "SeparatrixCalculator::isNearTransition: true" << std::endl;
    std::cout << "gamma = " << gamma << ", gt = " << gt;
    std::cout << ", Tc(" << Tc << ") > t(" << abs(gamma -gt)/dgamma << ")" << std::endl;
    */
    return true;
  } else {
    /*
    std::cout << "SeparatrixCalculator::isNearTransition: false" << std::endl;
    std::cout << "gamma = " << gamma << ", gt = " << gt;
    std::cout << ", Tc(" << Tc << ") < t(" << abs(gamma -gt)/dgamma << ")" << std::endl;
    */
    return false;
  }

}

double UAL::SeparatrixCalculator::getCtMax(double et)
{
  double charge = m_ba.getCharge();
  double energy = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (energy*energy - m*m)/energy/energy;
  double v0byc  = sqrt(v0byc2);
  double gamma  = energy/m;
  double gamma2 = gamma*gamma;

  double V      = m_rfTracker.getV();
  double harmon = m_rfTracker.getHarmon();
  double lag    = m_rfTracker.getLag();

  double eta0    = m_alpha0 - 1./gamma2;

  double phaseS = lag*(2.*UAL::pi);
  
  double revFreq= UAL::clight*v0byc/m_suml;
  double omega  = 2*UAL::pi*revFreq;
  double omega2 = omega*omega;

  double A = et;

  double dphi = 2*harmon*harmon*harmon*omega2*eta0;
  dphi       /= UAL::pi*1.0e+18*energy*v0byc2*charge*V*std::cos(phaseS);
  dphi        = std::abs(dphi);
  dphi        = std::pow(dphi, 1./4.);
  dphi       *= std::sqrt(A);

  // std::cout << "dphi = " << dphi << std::endl;

  double ct2phi = (2.0*UAL::pi*harmon)*(v0byc/m_suml);

  double ctMax = dphi/ct2phi;
  return ctMax;
}

double UAL::SeparatrixCalculator::getDeMax()
{
  double lag    = m_rfTracker.getLag();
  double phaseS = lag*(2.0*UAL::pi);

  double de2   = getHsep(UAL::pi - phaseS); 

  // std::cout << "de2 = " << de2 << std::endl;
  if(de2 < 0.0) {
    de2   = getHsep(phaseS); 
    // std::cout << "de2 = " << de2 << std::endl;
  }
  if(de2 < 0.0) {
    return 0.0;
  }
  
  return sqrt(de2);
}

double UAL::SeparatrixCalculator::getDeMax(double et)
{
  double charge = m_ba.getCharge();
  double energy = m_ba.getEnergy();
  double m      = m_ba.getMass();
  double v0byc2 = (energy*energy - m*m)/energy/energy;
  double v0byc  = sqrt(v0byc2);
  double p2     = energy*energy - m*m;
  double p      = sqrt(p);
  double gamma  = energy/m;
  double gamma2 = gamma*gamma;

  double V      = m_rfTracker.getV();
  double harmon = m_rfTracker.getHarmon();
  double lag    = m_rfTracker.getLag();

  double eta0    = m_alpha0 - 1./gamma2;

  double phaseS = lag*(2.*UAL::pi);
  
  double revFreq= UAL::clight*v0byc/m_suml;
  double omega  = 2*UAL::pi*revFreq;
  double omega2 = omega*omega;

  double A = et;

  double dw  = UAL::pi*1.0e+18*energy*v0byc2*charge*V*cos(phaseS);
  dw        /= 2*harmon*harmon*harmon*omega2*eta0;
  dw         = abs(dw);
  dw         = pow(dw, 1./4.);
  dw        *= sqrt(A)/UAL::pi;

  // std::cout << "dw = " << dw << std::endl;

  double deMax  = dw*harmon*omega/p/1.e+9;

  return deMax;
}

