// Library       : 
// File          : 
// Copyright     : see Copyright file
// Author        : 
// C++ version   :

#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/BasicTracker.hh"
#include "bendAndMultipole.hh"

THINSPIN::bendAndMultipoleAlgorithm<double, PAC::Position> THINSPIN::bendAndMultipole::s_algorithm;

THINSPIN::bendAndMultipole::bendAndMultipole()
  : TEAPOT::BasicTracker()
{
}

THINSPIN::bendAndMultipole::bendAndMultipole(const THINSPIN::bendAndMultipole& dt)
  : TEAPOT::BasicTracker(dt)
{
  m_data = dt.m_data;
  m_mdata = dt.m_mdata;
}

THINSPIN::bendAndMultipole::~bendAndMultipole()
{
}

UAL::PropagatorNode* THINSPIN::bendAndMultipole::clone()
{
  return new THINSPIN::bendAndMultipole(*this);
}

void THINSPIN::bendAndMultipole::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					       int is0, 
					       int is1,
					       const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void THINSPIN::bendAndMultipole::setLatticeElement(const PacLattElement& e)
{
  m_data.setLatticeElement(e);
  m_mdata.setLatticeElement(e);
}

void THINSPIN::bendAndMultipole::propagate(UAL::Probe& probe)
{
//std::cout << "JDT - enter void THINSPIN::bendAndMultipole::propagate(UAL::Probe& probe)\n";
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
//  s_algorithm.passBend(m_data, m_mdata, p, tmp, v0byc);
    s_algorithm.passBend(m_data, m_mdata, p, tmp, v0byc, ip);
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

THINSPIN::bendAndMultipoleRegister::bendAndMultipoleRegister()
{
  UAL::PropagatorNodePtr bendAndMultipolePtr((UAL::PropagatorNode*)new bendAndMultipole());
//UAL::PropagatorNodePtr bendAndMultipolePtr(                      new bendAndMultipole());
  UAL::PropagatorFactory::getInstance().add("THINSPIN::bendAndMultipole", bendAndMultipolePtr);
}

static THINSPIN::bendAndMultipoleRegister thebendAndMultipoleRegister;
