#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
//include         "ETEAPOT/Integrator/DriftTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DriftTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DipoleTracker.hh"
//#include "ETEAPOT/Integrator/DipoleTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

ETEAPOT_MltTurn::DriftAlgorithm<double, PAC::Position> ETEAPOT_MltTurn::DriftTracker::s_algorithm;
int ETEAPOT_MltTurn::DriftTracker::drft=0;

ETEAPOT_MltTurn::DriftTracker::DriftTracker()
  : ETEAPOT::BasicTracker()
{
}

ETEAPOT_MltTurn::DriftTracker::DriftTracker(const ETEAPOT_MltTurn::DriftTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
}

ETEAPOT_MltTurn::DriftTracker::~DriftTracker()
{
}

UAL::PropagatorNode* ETEAPOT_MltTurn::DriftTracker::clone()
{
  return new ETEAPOT_MltTurn::DriftTracker(*this);
}

void ETEAPOT_MltTurn::DriftTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
}

void ETEAPOT_MltTurn::DriftTracker::propagate(UAL::Probe& probe)
{
  // std::cout << "File " << __FILE__ << " line " << __LINE__
  // << " method void ETEAPOT_MltTurn::DriftTracker::propagate(UAL::Probe& probe)\n";

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
    s_algorithm.passDriftPlusPostProcess( ip, m_l, p, tmp, v0byc, ETEAPOT_MltTurn::DipoleTracker::m_m, drft );
  }
drft++;

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
