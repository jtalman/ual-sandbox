// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "DipoleTracker.hh"

#include <cstdlib>

algorithm<double, PAC::Position> ETEAPOT::DipoleTracker::s_algorithm;
double ETEAPOT::DipoleTracker::m_m;
   int ETEAPOT::DipoleTracker::bend=0;

ETEAPOT::DipoleTracker::DipoleTracker()
  : ETEAPOT::BasicTracker()
{
}

ETEAPOT::DipoleTracker::DipoleTracker(const ETEAPOT::DipoleTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
  m_data = dt.m_data;
  m_edata = dt.m_edata;

/*
  string line;
  ifstream m_m;
  m_m.open ("m_m");
  getline (m_m,line);
  ETEAPOT::DipoleTracker::m_m = atof( line.c_str() );
  m_m.close();
*/
}

ETEAPOT::DipoleTracker::~DipoleTracker()
{
}

UAL::PropagatorNode* ETEAPOT::DipoleTracker::clone()
{
  return new ETEAPOT::DipoleTracker(*this);
}

void ETEAPOT::DipoleTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					       int is0, 
					       int is1,
					       const UAL::AttributeSet& attSet)
{
std::cout << "TDJ - server side - File " << __FILE__ << " line " << __LINE__ << " enter method void ETEAPOT::DipoleTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,int is0,int is1,const UAL::AttributeSet& attSet)\n";
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void ETEAPOT::DipoleTracker::setLatticeElement(const PacLattElement& e)
{
std::cout << "TDJ - server side - File " << __FILE__ << " line " << __LINE__ << " enter method void ETEAPOT::DipoleTracker::setLatticeElement(const PacLattElement& e)\n";
std::cout << "e.getName() " << e.getName() << " e.getPosition() " << e.getPosition() << "\n";
  m_data.m_m=ETEAPOT::DipoleTracker::m_m;
  m_data.setLatticeElement(e);
  m_edata.setLatticeElement(e);
}

void ETEAPOT::DipoleTracker::propagate(UAL::Probe& probe)
{
std::cout << "TDJ - server side - File " << __FILE__ << " line " << __LINE__ << " enter method void ETEAPOT::DipoleTracker::propagate(UAL::Probe& probe) - new algorithm\n";
  UAL::AcceleratorNode fAN = getFrontAcceleratorNode();

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  const PAC::BeamAttributes cba = ba;

//  -------------------------------------------- //
//#include "printPropagateInfo.h"
  double e0 = ba.getEnergy();
  double m0 = ba.getMass();
  double t0 = ba.getElapsedTime();
  double oldT = t0;
  double p0   = sqrt(e0*e0 - m0*m0);
//  -------------------------------------------- //

  double v0byc = p0/e0;

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
              #include "verboseBlock.h"
//  if(ip==0){#include "verboseBlock.h"}
    s_algorithm.passEntry(0, m_edata, p, 0, ETEAPOT::DipoleTracker::m_m );
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passBend( ip, m_data, m_edata, p, tmp, v0byc, cba, bend );
    s_algorithm.passExit(0, m_edata, p, 0, ETEAPOT::DipoleTracker::m_m );
    // testAperture(p);
  }
bend++;

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
