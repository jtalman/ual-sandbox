// Library       : ACCSIM
// File          : ACCSIM/Collimator/FermiScatter.cc
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#include <math.h>
#include "ACCSIM/Collimator/FermiScatter.hh"

ACCSIM::FermiScatter::FermiScatter()
  : m_z(1.0), m_rev_vp(1.0), m_radlength_factor(1.0)
{
}

ACCSIM::FermiScatter::~FermiScatter()
{
}

void ACCSIM::FermiScatter::setMaterial(double, double, double, double radLength){
  m_radlength_factor = 1./sqrt(radLength);
} 

double ACCSIM::FermiScatter::getRadLength() const {
  return 1./(m_radlength_factor*m_radlength_factor);
}

void ACCSIM::FermiScatter::setBeam(double m, double e, double charge)
{
  double p2 = e*e - m*m;

  m_rev_vp = e/p2;
  m_z = charge;
}

double ACCSIM::FermiScatter::getRmsAngle() const
{
  return 0.0136*m_radlength_factor*m_z*m_rev_vp;
}

void ACCSIM::FermiScatter::update(PAC::Particle& part, double s, int& iseed)
{

  PAC::Position& pos = part.getPosition();

  double theta0 = getRmsAngle();
  double thetaS = theta0*sqrt(s);

  double z1, z2, dz, dpz;

  // x-plane

  z2 = m_gaussGenerator.getNumber(iseed);
  z1 = m_gaussGenerator.getNumber(iseed);
  dz = s*pos.getPX() + s*thetaS/2.0*(z1/sqrt(3.0) + z2);
  dpz = thetaS*z2;

  double x  = pos.getX() + dz;
  double px = pos.getPX() + dpz;
  pos.setX(x);
  pos.setPX(px);

  // y-plane

  z2 = m_gaussGenerator.getNumber(iseed);
  z1 = m_gaussGenerator.getNumber(iseed);
  dz = s*pos.getPY() + s*thetaS/2.0*(z1/sqrt(3.0) + z2);
  dpz = thetaS*z2; 

  double y  = pos.getY() + dz;
  double py = pos.getPY() + dpz;
  pos.setY(y);
  pos.setPY(py);

}
