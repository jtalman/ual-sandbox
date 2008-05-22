// Library       : ACCSIM
// File          : ACCSIM/Collimator/NuclearInteraction.cc
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky

#include <math.h>
#include "ACCSIM/Base/Def.hh"
#include "ACCSIM/Collimator/NuclearInteraction.hh"


ACCSIM::NuclearInteraction::NuclearInteraction() 
{
  setMaterial(1., 1., 1.0, 0.0);
  setBeam(0.938, 0.938, 1.);
}

ACCSIM::NuclearInteraction::~NuclearInteraction()
{
}

void ACCSIM::NuclearInteraction::setBeam(double m, double energy, double)
{
  m_p = sqrt(energy*energy - m*m);
}

void ACCSIM::NuclearInteraction::setMaterial(double A, double Z, double rho, double)
{
  m_A = A;
  m_Z = Z;
  m_rho = rho;

  calculateCSs(A);
  calculateN(A, rho);
}

void ACCSIM::NuclearInteraction::update(PAC::Particle& part, double , int& iseed)
{
  double rnd = m_uGenerator.getNumber(iseed); 
   
  if(rnd > 0 && rnd < m_eP) { // Elastic reaction
    makeElasticScattering(part.getPosition(), iseed);
  } 
  else {                    // Inelastic reaction
    part.setFlag(1);
  }
}

void ACCSIM::NuclearInteraction::makeElasticScattering(PAC::Position& pos, int &iseed)
{
 
  // double term = sqrt(1./3.*pow(m_A, 2./3.));
  // double th0 = ACCSIM::pimass/(term*m_p);

  double term = m_p/ACCSIM::pimass;
  double b = 1./6.*pow(m_A, 2./3.)*term*term;
		     
  double rnd = m_uGenerator.getNumber(iseed);
  if(rnd == 0.0) rnd =  m_uGenerator.getNumber(iseed);
  
  double t = -log(rnd)/b;
  double th = sqrt(t);

  // 
  double sigran = 2.*ACCSIM::pi*m_uGenerator.getNumber(iseed);
  double px = pos.getPX() + th*cos(sigran);
  double py = pos.getPY() + th*sin(sigran);

  pos.setPX(px);
  pos.setPY(py);
}

double  ACCSIM::NuclearInteraction::getRlam() const {

  double rlam = 1.e+24/m_N/(m_sige + m_sigi);
  return rlam/100; // convert to meters
}

// ACCSIM target/nisetup after Igo et al.
double  ACCSIM::NuclearInteraction::getElasticCS() const{
  return m_sige; // barn
}

// ACCSIM target/nisetup after Williams
double  ACCSIM::NuclearInteraction::getInelasticCS() const{
  return m_sigi; // barn 
}

// ACCSIM target/nisetup after Igo et al.
void ACCSIM::NuclearInteraction::calculateCSs(double A) {

  double am13 = pow(A, -1./3.);
  double sigbar = 37.;  // elementary cross section per nucleon
  double sbar33 = sigbar - 33; 

  m_sigi = 0.044*pow(A, 0.69);
  m_sigi *= (1. + 0.039*am13*sbar33 - 0.0009*am13*sbar33*sbar33);

  double sigt = 0;
  if(m_A < 70) {
    sigt = 0.047*pow(A, 0.82);
  } else {
    sigt = 0.089*pow(A, 0.67);
  }

  m_sige = sigt - m_sigi;
  m_eP = m_sige/sigt;
}

// ACCSIM target/nisetup 
void ACCSIM::NuclearInteraction::calculateN(double A, double rho) {
  m_N = ACCSIM::nAvogadro*rho/A;
}


