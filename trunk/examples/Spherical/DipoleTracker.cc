// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "DipoleTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

newDipoleAlgorithm<double, PAC::Position> s_algorithm;

 double ETEAPOT::DipoleTracker::xS[1000];
 double ETEAPOT::DipoleTracker::yS[1000];
 double ETEAPOT::DipoleTracker::zS[1000];
 char   ETEAPOT::DipoleTracker::nS[1000][100];
 int    ETEAPOT::DipoleTracker::maxSurvey;

ETEAPOT::DipoleTracker::DipoleTracker()
  : ETEAPOT::BasicTracker()
{
 std::cout << "File " << __FILE__ << " line " << __LINE__ << " enter method ETEAPOT::DipoleTracker::DipoleTracker() : ETEAPOT::BasicTracker()\n";
 std::ifstream inFile; 
 char index[14];  // One extra for null char.
 char name[14];  // One extra for null char.

 int i=0;

 inFile.open("Survey", std::ios::in);

 if (!inFile) {
  std::cout << "Can't open input file " << "Survey" << std::endl;
  exit(1);
 }

 while (inFile >> index >> ETEAPOT::DipoleTracker::xS[i] >> ETEAPOT::DipoleTracker::yS[i] >> ETEAPOT::DipoleTracker::zS[i] >> ETEAPOT::DipoleTracker::nS[i]) {
  ETEAPOT::DipoleTracker::maxSurvey=i;
  i++;
 }
}

ETEAPOT::DipoleTracker::DipoleTracker(const ETEAPOT::DipoleTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
  m_data = dt.m_data;
  m_edata = dt.m_edata;
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
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void ETEAPOT::DipoleTracker::setLatticeElement(const PacLattElement& e)
{
  m_data.setLatticeElement(e);
  m_edata.setLatticeElement(e);
}

void ETEAPOT::DipoleTracker::propagate(UAL::Probe& probe)
{
std::cout << "TDJ - client side - File " << __FILE__ << " line " << __LINE__ << " enter method void ETEAPOT::DipoleTracker::propagate(UAL::Probe& probe)\n";
  UAL::AcceleratorNode fAN = getFrontAcceleratorNode();

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  const PAC::BeamAttributes cba = ba;
#include "printPropagateInfo.h"
  double v0byc = p0/e0;

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
#include "verboseBlock.h"
    s_algorithm.passEntry(m_edata, p);
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
std::cout << "File " << __FILE__ << " line " << __LINE__ << " about to  s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc);\n";
    s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc, cba);
//                      [      original interface      ] [central orbit ]
std::cout << "File " << __FILE__ << " line " << __LINE__ << " back from s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc, e0, p0, m0);\n";
    s_algorithm.passExit(m_edata, p);
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










