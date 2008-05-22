


#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"

#include "TEAPOT/StringSC/StringSCSolver.hh"
#include "TEAPOT/StringSC/MltStringSCKick.hh"

TEAPOT::MagnetAlgorithm<double, PAC::Position> TEAPOT::MltStringSCKick::s_algorithm;

TEAPOT::MltStringSCKick::MltStringSCKick()
  : TEAPOT::BasicTracker()
{
  initialize();
}

TEAPOT::MltStringSCKick::MltStringSCKick(const TEAPOT::MltStringSCKick& mt)
  : TEAPOT::BasicTracker(mt)
{
  copy(mt);
}

TEAPOT::MltStringSCKick::~MltStringSCKick()
{
}

UAL::PropagatorNode* TEAPOT::MltStringSCKick::clone()
{
  return new TEAPOT::MltStringSCKick(*this);
}

void TEAPOT::MltStringSCKick::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					    int is0, 
					    int is1,
					    const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void TEAPOT::MltStringSCKick::setLatticeElement(const PacLattElement& e)
{
  // length
  // m_l = e.getLength();

  // ir
  m_ir = e.getN();

  m_mdata.setLatticeElement(e);

}

void TEAPOT::MltStringSCKick::propagateSimpleElement(PAC::Bunch& bunch, double v0byc)
{

  PAC::Position tmp;

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  // double v0byc = p0/e0;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

      s_algorithm.makeVelocity(p, tmp, v0byc);
      s_algorithm.makeRV(p, tmp, e0, p0, m0);

      s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      s_algorithm.applyMltKick(m_mdata, 1., p); 

      // s_algorithm.makeVelocity(p, tmp, v0byc);
  }

  // string sc kick
  TEAPOT::StringSCSolver& scSolver = TEAPOT::StringSCSolver::getInstance();

  double L  = m_l;
  double Ri = 0.0; // m_angle*bendfac/m_l;

  int sRi = 1;
  if ( Ri < 0 ) sRi = -1; 

  std::cout << m_name << " mlt (simple) , L = " << L << ", Ri = " << Ri << ", sRi = " << sRi << std::endl;

  Ri = fabs(Ri);
  scSolver.propagate(bunch, Ri, sRi, L); // string sc solver

  e0 = ba.getEnergy();
  m0 = ba.getMass();
  p0 = sqrt(e0*e0 - m0*m0);
  // v0byc = p0/e0;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);
    s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
  } 

}

void TEAPOT::MltStringSCKick::propagateComplexElement(PAC::Bunch& bunch, double v0byc)
{

  PAC::Position tmp;

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  // v0byc = p0/e0;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    double rIr = 1./m_ir;
    double rkicks = 0.25*rIr;

    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);

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

  }

  // string sc kick
  TEAPOT::StringSCSolver& scSolver = TEAPOT::StringSCSolver::getInstance();

  double L  = m_l;
  double Ri = 0.0; // m_angle*bendfac/m_l;

  int sRi = 1;
  if ( Ri < 0 ) sRi = -1; 

  std::cout << m_name << " mlt (complex) , L = " << L << ", Ri = " << Ri << ", sRi = " << sRi << ", ir = " << m_ir << std::endl;
  
  /*
  std::cout << "before sc solver " << std::endl;
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
  std::cout << "after sc solver " << std::endl;
  for(int i =0; i < bunch.size(); i++){
    PAC::Position p = bunch[i].getPosition();
    std::cout << i << " " 
	      << p[0] << " " << p[1] << " " 
	      << p[2] << " " << p[3] << " " 
	      << p[4] << " " << p[5] << std::endl;
  }
  */

}

void TEAPOT::MltStringSCKick::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  double oldT = ba.getElapsedTime();

  PAC::Position tmp;

  // Entry
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    s_algorithm.passEntry(m_mdata, p);
    
    // s_algorithm.makeVelocity(p, tmp, v0byc);
    // s_algorithm.makeRV(p, tmp, e0, p0, m0);
  }

  if(!m_ir) propagateSimpleElement(bunch, v0byc);
  else propagateComplexElement(bunch, v0byc);


  // Exit
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    s_algorithm.passExit(m_mdata, p);
  }

  checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);  

}

void TEAPOT::MltStringSCKick::initialize()
{
  // m_l = 0.0;
  m_ir = 0.0;
}

void TEAPOT::MltStringSCKick::copy(const TEAPOT::MltStringSCKick& mt)
{
  // m_l   = mt.m_l;
  m_ir  = mt.m_ir;

  m_mdata = mt.m_mdata;
}

TEAPOT::MltStringSCKickRegister::MltStringSCKickRegister()
{
  UAL::PropagatorNodePtr nodePtr(new TEAPOT::MltStringSCKick());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::MltStringSCKick", nodePtr);
}



static TEAPOT::MltStringSCKickRegister theTeapotMltStringSCKickRegister; 








