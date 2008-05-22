
#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/StringSC/StringSCSolver.hh"
#include "TEAPOT/StringSC/DriftStringSCKick.hh"

TEAPOT::DriftAlgorithm<double, PAC::Position> TEAPOT::DriftStringSCKick::s_algorithm;

TEAPOT::DriftStringSCKick::DriftStringSCKick()
  : TEAPOT::BasicTracker()
{
}

TEAPOT::DriftStringSCKick::DriftStringSCKick(const TEAPOT::DriftStringSCKick& dt)
  : TEAPOT::BasicTracker(dt)
{
}

TEAPOT::DriftStringSCKick::~DriftStringSCKick()
{
}

UAL::PropagatorNode* TEAPOT::DriftStringSCKick::clone()
{
  return new TEAPOT::DriftStringSCKick(*this);
}


void TEAPOT::DriftStringSCKick::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;

   m_name = lattice[is0].getName();
   std::cout << "Set Drift Kick " << m_name << std::endl;
}

void TEAPOT::DriftStringSCKick::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double oldT = ba.getElapsedTime();

  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);

  double v0byc = p0/e0;

  PAC::Position tmp;

  /*
  std::cout << "before drift " << v0byc << std::endl;
  for(int i =0; i < bunch.size(); i++){
    PAC::Position p = bunch[i].getPosition();
    std::cout << i << " " 
	      << p[0] << " " << p[1] << " " 
	      << p[2] << " " << p[3] << " " 
	      << p[4] << " " << p[5] << std::endl;
  }
  */

  // half drift
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passDrift(m_l/2, p, tmp, v0byc);
  }

    // string sc kick
    TEAPOT::StringSCSolver& scSolver = TEAPOT::StringSCSolver::getInstance();

    double L  = m_l;
    double Ri = 0.0; // m_angle*bendfac/m_l;

    int sRi = 1;
    if ( Ri < 0 ) sRi = -1; 

    std::cout << m_name << " drift kick , L = " << L << ", Ri = " << Ri << ", sRi = " << sRi << std::endl;

    /*
    std::cout << "before solver " << std::endl;
    for(int i =0; i < bunch.size(); i++){
      PAC::Position p = bunch[i].getPosition();
      std::cout << i << " " 
		<< p[0] << " " << p[1] << " " 
		<< p[2] << " " << p[3] << " " 
		<< p[4] << " " << p[5] << std::endl;
    }
    */

    Ri = fabs(Ri);
    scSolver.propagate(bunch, Ri, sRi, L); 


    /*
    std::cout << "after solver " << std::endl;
    for(int i =0; i < bunch.size(); i++){
      PAC::Position p = bunch[i].getPosition();
      std::cout << i << " " 
		<< p[0] << " " << p[1] << " " 
		<< p[2] << " " << p[3] << " " 
		<< p[4] << " " << p[5] << std::endl;
    }
    */

    // half drift
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passDrift(m_l/2, p, tmp, v0byc);
  }

  /*
    std::cout << "after drift " << std::endl;
    for(int i =0; i < bunch.size(); i++){
      PAC::Position p = bunch[i].getPosition();
      std::cout << i << " " 
		<< p[0] << " " << p[1] << " " 
		<< p[2] << " " << p[3] << " " 
		<< p[4] << " " << p[5] << std::endl;
    }
  */

    checkAperture(bunch);

    // ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);
}

TEAPOT::DriftStringSCKickRegister::DriftStringSCKickRegister()
{
  UAL::PropagatorNodePtr nodePtr(new TEAPOT::DriftStringSCKick());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DriftStringSCKick", nodePtr);
}



static TEAPOT::DriftStringSCKickRegister theTeapotDriftStringSCKickRegister; 
