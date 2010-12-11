// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DriftTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/DriftTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

ETEAPOT::DriftAlgorithm<double, PAC::Position> ETEAPOT::DriftTracker::s_algorithm;

ETEAPOT::DriftTracker::DriftTracker()
  : ETEAPOT::BasicTracker()
{
}

ETEAPOT::DriftTracker::DriftTracker(const ETEAPOT::DriftTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
}

ETEAPOT::DriftTracker::~DriftTracker()
{
}

UAL::PropagatorNode* ETEAPOT::DriftTracker::clone()
{
  return new ETEAPOT::DriftTracker(*this);
}


void ETEAPOT::DriftTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
}

void ETEAPOT::DriftTracker::propagate(UAL::Probe& probe)
{
std::cout << "File " << __FILE__ << " line " << __LINE__ << " method void ETEAPOT::DriftTracker::propagate(UAL::Probe& probe)\n";
//std::cout << "Hello ETEAPOT::DriftTracker " << std::endl;

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
    std::cout << "m_i0 " << m_i0 << "\n";
    std::cout << "m_i1 " << m_i1 << "\n";
    std::cout << "m_l " << m_l << "\n";
    std::cout << "m_n " << m_n << "\n";
    std::cout << "m_s " << m_s << "\n";
    std::cout << "m_name " << m_name << "\n";
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

