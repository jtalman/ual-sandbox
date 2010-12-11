// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/DipoleTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

ETEAPOT::DipoleAlgorithm<double, PAC::Position> ETEAPOT::DipoleTracker::s_algorithm;

 double ETEAPOT::DipoleTracker::xS[1000];
 double ETEAPOT::DipoleTracker::yS[1000];
 double ETEAPOT::DipoleTracker::zS[1000];
 char   ETEAPOT::DipoleTracker::nS[1000][100];
 int    ETEAPOT::DipoleTracker::maxSurvey;

ETEAPOT::DipoleTracker::DipoleTracker()
  : ETEAPOT::BasicTracker()
{
 std::cout << "File " << __FILE__ << " line " << __LINE__ << " method ETEAPOT::DipoleTracker::DipoleTracker() : ETEAPOT::BasicTracker()\n";
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
std::cout << "File " << __FILE__ << " line " << __LINE__ << " method void ETEAPOT::DipoleTracker::propagate(UAL::Probe& probe)\n";

//  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
//  Teapot* teapot = optics.m_teapot;
//  PacSurveyData surveyData;
//  TeapotElement te = teapot->element(m_i0);
//  std::string nameInput;
//  nameInput=te.getDesignName();
//  std::cout << "nameInput=te.getDesignName() " << nameInput << "\n";
  std::cout << "m_i0 " << m_i0 << "\n";
  std::cout << "m_i1 " << m_i1 << "\n";
  std::cout << "m_l " << m_l << "\n";
  std::cout << "m_n " << m_n << "\n";
  std::cout << "m_s " << m_s << "\n";
  std::cout << "m_name " << m_name << "\n";
  std::cout << "m_data.m_l " << m_data.m_l << "\n";
  std::cout << "m_data.m_ir " << m_data.m_ir << "\n";
  std::cout << "m_data.m_angle " << m_data.m_angle << "\n";
  std::cout << "R = m_data.m_l/m_data.m_angle " << m_data.m_l/m_data.m_angle << "\n";
  std::cout << "m_data.m_atw00 " << m_data.m_atw00 << "\n";
  std::cout << "m_data.m_atw01 " << m_data.m_atw01 << "\n";
  std::cout << "m_data.m_btw00 " << m_data.m_btw00 << "\n";
  std::cout << "m_data.m_btw01 " << m_data.m_btw01 << "\n";

//teapot->survey(surveyData,m_i0,m_i0+1);
/*
  double xLS = surveyData.survey().x();
  double yLS = surveyData.survey().y();
  double zLS = surveyData.survey().z();
  std::cout << "xLS      " << xLS      << " yLS      " << yLS      << " zLS      " << zLS      << "\n";
*/
std::cout << "       xS[m_i0] " << ETEAPOT::DipoleTracker::xS[m_i0] << " member yS[m_i0] " << ETEAPOT::DipoleTracker::yS[m_i0] << " member zS[m_i0] " << ETEAPOT::DipoleTracker::zS[m_i0] << " member nS[m_i0] " << ETEAPOT::DipoleTracker::nS[m_i0] << "\n";
PacSurvey survey=m_data.m_slices[0].survey();
std::cout << "member xS[m_i0] " << survey.x() << " member yS[m_i0] " << survey.y() << " member zS[m_i0] " << survey.z() << "\n";

  UAL::AcceleratorNode fAN = getFrontAcceleratorNode();

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
#include "verboseBlock.h"
    s_algorithm.passEntry(m_edata, p);
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
std::cout << "File " << __FILE__ << " line " << __LINE__ << " about to  s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc);\n";
    s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc);
std::cout << "File " << __FILE__ << " line " << __LINE__ << " back from s_algorithm.passBend(m_data, m_edata, p, tmp, v0byc);\n";
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










