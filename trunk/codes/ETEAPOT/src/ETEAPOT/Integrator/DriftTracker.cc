// Library       : TEAPOT
// File          : TEAPOT/Integrator/DriftTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/DriftTracker.hh"

TEAPOT::DriftAlgorithm<double, PAC::Position> TEAPOT::DriftTracker::s_algorithm;

TEAPOT::DriftTracker::DriftTracker()
  : TEAPOT::BasicTracker()
{
}

TEAPOT::DriftTracker::DriftTracker(const TEAPOT::DriftTracker& dt)
  : TEAPOT::BasicTracker(dt)
{
}

TEAPOT::DriftTracker::~DriftTracker()
{
}

UAL::PropagatorNode* TEAPOT::DriftTracker::clone()
{
  return new TEAPOT::DriftTracker(*this);
}


void TEAPOT::DriftTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
}

void TEAPOT::DriftTracker::propagate(UAL::Probe& probe)
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
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passDrift(m_l, p, tmp, v0byc);
  }

  /*
  std::cout << "after drift " << m_name << std::endl;
  for(int i =0; i < bunch.size(); i++){
    PAC::Position p = bunch[i].getPosition();
    std::cout << i << " " 
	      << p[0] << " " << p[1] << " " 
	      << p[2] << " " << p[3] << " " 
	      << p[4] << " " << p[5] << std::endl;
  }
  */


  checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);
}

