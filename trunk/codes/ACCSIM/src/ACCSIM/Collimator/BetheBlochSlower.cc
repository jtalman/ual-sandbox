// Library       : ACCSIM
// File          : ACCSIM/Collimator/BetheBlochSlower.cc
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#include <math.h>
#include "ACCSIM/Collimator/BetheBlochSlower.hh"

ACCSIM::BetheBlochSlower::BetheBlochSlower(){
  setMaterial(1., 1., 1., 1.0);
  setBeam(ACCSIM::pmass, ACCSIM::pmass, 1.0);
}

ACCSIM::BetheBlochSlower::~BetheBlochSlower(){
}

void ACCSIM::BetheBlochSlower::setMaterial(double A, double Z, double rho, double){
  m_A   = A;
  m_Z   = Z;
  m_rho = rho;
}

void ACCSIM::BetheBlochSlower::setBeam(double m, double energy, double){
  m_m0   = m;
  m_energy = energy;
  m_beta = sqrt(m_energy*m_energy - m_m0*m_m0)/m_energy; // beta;
}

void ACCSIM::BetheBlochSlower::update(PAC::Particle& part, double l, int& iseed)
{
  PAC::Position& pos = part.getPosition();

  double eloss = getMeanEnergyLoss(pos)*l;
  double p0 = m_energy*m_beta;
  double de = pos.getDE()*p0 - eloss;

  //added goodies for the time delay
  //this uses the energy deviation before scattering to estimate the time delay
  // can be easily adjusted to make it after scattering
  double t0=1.0/sqrt(1.0+(pos[5]+2.0/m_beta)*pos[5]-pos[1]*pos[1]-pos[3]*pos[3]);
  double p1=pos[1]*pos[1]/(t0*t0)+pos[3]*pos[3]/(t0*t0);
  double p4=0.5*(1.0+sqrt(1.0+p1));
  double E=m_energy+p0*pos[5];
  double cbyrv=E/sqrt(E*E-m_m0*m_m0);
  double p2=(1.0/m_beta-cbyrv)*l;

  p1=p1/p4*cbyrv*l/2.0;

  pos.setDE(de/p0);
  pos[4] -=p1;
  pos[4] +=p2;
  
}

double  ACCSIM::BetheBlochSlower::getEnergyLoss(double meanLoss, double sigLoss, double l, int& iseed)
{
  double r = m_uGenerator.getNumber(iseed);
  double de = meanLoss*l + sigLoss*sqrt(l)*getGaussIn(r);
  return de;
}

double  ACCSIM::BetheBlochSlower::getMeanEnergyLoss(PAC::Position& pos)
{
  double p = m_energy*m_beta;
  double energy = m_energy + pos.getDE()*p;

  double beta = sqrt(energy*energy - m_m0*m_m0)/energy;
  double beta2 = beta*beta;
  double gamma2 = 1./(1. - beta2);
  double gamma = sqrt(gamma2); 

  double me_by_m0 = ACCSIM::emass/m_m0;
  double emax = 2*ACCSIM::emass*beta2*gamma2/(1. + 2.*gamma*me_by_m0 + me_by_m0*me_by_m0);

  // double de  = getXi();
  double de  = 0.30058*m_Z*ACCSIM::emass/(m_A*beta*beta);
  de *= m_rho*100;

  double VI = 10.e-9*m_Z; // mean excitation energy
  de *= log(2.*ACCSIM::emass*beta2*emax*gamma2/VI/VI) - 2.*beta2;

  return de;
}

double  ACCSIM::BetheBlochSlower::getMeanEnergyLoss()
{
  double beta2 = m_beta*m_beta;
  double gamma2 = 1./(1. - beta2);

  double emax = getEmax();

  double de  = getXi();

  double VI = 10.e-9*m_Z; // mean excitation energy
  de *= log(2.*ACCSIM::emass*beta2*emax*gamma2/VI/VI) - 2.*beta2;
  
  return de;
}

double  ACCSIM::BetheBlochSlower::getVarianceEnergyLoss()
{
  double sig2 = getXi()*getEmax()*(1 - m_beta*m_beta/2.0);  
  return sqrt(sig2);
}

double ACCSIM::BetheBlochSlower::getEnergyLossParameter()
{
  double xi = getXi();
  double emax = getEmax();
  return xi/emax;
}

double ACCSIM::BetheBlochSlower::getEmax()
{
  double beta2 = m_beta*m_beta;
  double gamma2 = 1./(1. - beta2);
  double gamma = sqrt(gamma2);  

  double me_by_m0 = ACCSIM::emass/m_m0;

  double emax = 2*ACCSIM::emass*beta2*gamma2/(1. + 2.*gamma*me_by_m0 + me_by_m0*me_by_m0);
  return emax;
}

double ACCSIM::BetheBlochSlower::getXi()
{
  double de = 0.30058*m_Z*ACCSIM::emass/(m_A*m_beta*m_beta);
  return de*m_rho*100; // Gev/m
}


double ACCSIM::BetheBlochSlower::getGaussIn(double r)
{
  return 0.0;
}
