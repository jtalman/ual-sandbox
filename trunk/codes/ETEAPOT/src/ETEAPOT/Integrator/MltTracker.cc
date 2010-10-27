// Library       : TEAPOT
// File          : TEAPOT/Integrator/MltTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/MltTracker.hh"

TEAPOT::MagnetAlgorithm<double, PAC::Position> TEAPOT::MltTracker::s_algorithm;

TEAPOT::MltTracker::MltTracker()
  : TEAPOT::BasicTracker()
{
  initialize();
}

TEAPOT::MltTracker::MltTracker(const TEAPOT::MltTracker& mt)
  : TEAPOT::BasicTracker(mt)
{
  copy(mt);
}

TEAPOT::MltTracker::~MltTracker()
{
}

UAL::PropagatorNode* TEAPOT::MltTracker::clone()
{
  return new TEAPOT::MltTracker(*this);
}

void TEAPOT::MltTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					    int is0, 
					    int is1,
					    const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void TEAPOT::MltTracker::setLatticeElement(const PacLattElement& e)
{
  // length
  // m_l = e.getLength();

  // ir
  m_ir = e.getN();

  m_mdata.setLatticeElement(e);

}

void TEAPOT::MltTracker::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  double oldT = ba.getElapsedTime();

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    s_algorithm.passEntry(m_mdata, p);

    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);

    // Simple Element

    if(!m_ir){
      s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      s_algorithm.applyMltKick(m_mdata, 1., p); 
      s_algorithm.makeVelocity(p, tmp, v0byc);
      s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      s_algorithm.passExit(m_mdata, p); 
      continue;
    } 

    // Complex Element


    double rIr = 1./m_ir;
    double rkicks = 0.25*rIr;

    int counter = 0;
    for(int i = 0; i < m_ir; i++){
      for(int is = 0; is < 4; is++){
	counter++;
	s_algorithm.passDrift(m_l*s_steps[is]*rIr, p, tmp, v0byc);
	s_algorithm.applyMltKick(m_mdata, rkicks, p); 
	s_algorithm.makeVelocity(p, tmp, v0byc);	
      }
      counter++;
      s_algorithm.passDrift(m_l*s_steps[4]*rIr, p, tmp, v0byc); 
    }

    s_algorithm.passExit(m_mdata, p);  
    // testAperture(p);
  }

  /*
  std::cout << "after quadrupole " << m_name << std::endl;
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

void TEAPOT::MltTracker::initialize()
{
  // m_l = 0.0;
  m_ir = 0.0;
}

void TEAPOT::MltTracker::copy(const TEAPOT::MltTracker& mt)
{
  // m_l   = mt.m_l;
  m_ir  = mt.m_ir;

  m_mdata = mt.m_mdata;
}










