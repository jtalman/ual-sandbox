// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/DipoleTracker.hh"

TEAPOT::DipoleAlgorithm<double, PAC::Position> TEAPOT::DipoleTracker::s_algorithm;

TEAPOT::DipoleTracker::DipoleTracker()
  : TEAPOT::BasicTracker()
{
}

TEAPOT::DipoleTracker::DipoleTracker(const TEAPOT::DipoleTracker& dt)
  : TEAPOT::BasicTracker(dt)
{
  m_data = dt.m_data;
  m_mdata = dt.m_mdata;
}

TEAPOT::DipoleTracker::~DipoleTracker()
{
}

UAL::PropagatorNode* TEAPOT::DipoleTracker::clone()
{
  return new TEAPOT::DipoleTracker(*this);
}

void TEAPOT::DipoleTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					       int is0, 
					       int is1,
					       const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void TEAPOT::DipoleTracker::setLatticeElement(const PacLattElement& e)
{
  m_data.setLatticeElement(e);
  m_mdata.setLatticeElement(e);
}

void TEAPOT::DipoleTracker::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double oldT = ba.getElapsedTime();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
    s_algorithm.passEntry(m_mdata, p);
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passBend(m_data, m_mdata, p, tmp, v0byc);
    s_algorithm.passExit(m_mdata, p);  
    // testAperture(p);
  }

  /*
  std::cout << "after dipole " << m_name << std::endl;
  for(int i =0; i < bunch.size(); i++){
    PAC::Position p = bunch[i].getPosition();
    std::cout << i << " " 
	      << p[0] << " " << p[1] << " " 
	      << p[2] << " " << p[3] << " " 
	      << p[4] << " " << p[5] << std::endl;
  }
  */

  checkAperture(bunch);

  // Should be edited with the correct length for sbend and rbends
  ba.setElapsedTime(oldT + m_data.m_l/v0byc/UAL::clight);
}










