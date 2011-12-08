// Library       : ACCSIM
// File          : ACCSIM/Collimator/CollimatorTracker.cc
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

//#include <iostream>
#include <cmath>
#include <algorithm>
#include "UAL/APF/PropagatorFactory.hh"
#include "ACCSIM/Base/Def.hh"
#include "ACCSIM/Collimator/CollimatorTracker.hh"
#include "SMF/PacLattice.h"

double ACCSIM::CollimatorTracker::s_maxsize = 1.0; // 1 m
double ACCSIM::CollimatorTracker::s_dstep = 1.e-9; // 1 m

ACCSIM::CollimatorTracker::CollimatorTracker() 
{
  init();
}

ACCSIM::CollimatorTracker::CollimatorTracker(const ACCSIM::CollimatorTracker& tracker) 
{
  copy(tracker);
}

ACCSIM::CollimatorTracker::~CollimatorTracker(){
}

UAL::PropagatorNode* ACCSIM::CollimatorTracker::clone()
{
  return new ACCSIM::CollimatorTracker(*this);
}

void ACCSIM::CollimatorTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						   int i0, int i1, 
						   const UAL::AttributeSet& attSet)
{
  ACCSIM::BasicPropagator::setLatticeElements(sequence, i0, i1, attSet);

  const PacLattice& lattice     = (PacLattice&) sequence;
  m_length = lattice[i0].getLength();
}

void ACCSIM::CollimatorTracker::propagate(UAL::Probe& b){

  std::cout << "ACCSIM::CollimatorTracker::propagate - propagating" << std::endl;
  std::cout << "A = " << m_A << std::endl;

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
 
  E0 = bunch.getBeamAttributes().getEnergy();
  m0 = bunch.getBeamAttributes().getMass();

  // int iseed  = -1;

  m_scatter.setBeam(m0, E0, bunch.getBeamAttributes().getCharge());
  m_stopper.setBeam(m0, E0, bunch.getBeamAttributes().getCharge());
  m_nuclearInteraction.setBeam(m0, E0, bunch.getBeamAttributes().getCharge());

  p0= sqrt(E0*E0-m0*m0);
  v0byc = p0/E0;

  for(int i = 0; i < bunch.size(); i++){

      PAC::Particle& particle = bunch[i];
      if(particle.getFlag() == 1 /* lost particle */ ) continue;
      update(particle, m_length, m_iseed); 
  }

}

void ACCSIM::CollimatorTracker::setLength(double l){
  m_length = l;
}

double ACCSIM::CollimatorTracker::getLength() const {
  return m_length;
}

void ACCSIM::CollimatorTracker::setMaterial(double A, double Z, double rho, double radlength){

  m_A = A;
  m_Z = Z;
  m_rho = rho;
  m_radlength = radlength;

  m_nuclearInteraction.setMaterial(A, Z, rho, radlength);
  m_stopper.setMaterial(A, Z, rho, radlength);
  m_scatter.setMaterial(A, Z, rho, radlength);
}

void ACCSIM::CollimatorTracker::setAperture(int shape, double a, double b){

  switch (shape){
  case ELLIPSE :
  case RECTANGLE :
    m_shape = shape;
    m_a = a > 0 ? a : s_maxsize;
    m_b = b > 0 ? b : s_maxsize;
    break;
  case XFLAT :
  case YFLAT :
    m_shape = shape;
    if (a>b) std::swap(a,b); //ensures that a<b
    m_a = a;  //left edge for X and upper edge for Y
    m_b = b;  //right edge for X and lower edge for Y
    break;
  default :  
    break;
  }

}

void ACCSIM::CollimatorTracker::setSeed(int iseed)
{
  m_iseed = iseed;
}

int ACCSIM::CollimatorTracker::getSeed() const
{
  return m_iseed;
}

int ACCSIM::CollimatorTracker::getLostParticles() const
{
  return m_nLosts;
}



void ACCSIM::CollimatorTracker::update(PAC::Particle& particle, double l, int& iseed)
{
  PAC::Position& pos = particle.getPosition();
  
  m_length = l;
  m_iseed = iseed;
  
  double at = 0.0;
  double rlam = m_nuclearInteraction.getRlam();
  
  while(at < m_length) {
    bool apFlag = checkAperture(pos);
    if(apFlag){
      trackThroughMaterial(particle, at, rlam);
    }
    else {
      trackThroughDrift(pos, at);
    }
  }   
}

bool ACCSIM::CollimatorTracker::checkAperture(const PAC::Position& pos) const {

  bool flag = false;

  double x = pos.getX();
  double y = pos.getY();

  switch(m_shape) {
  case ELLIPSE :
    x /= m_a;
    y /= m_b;
    if((x*x + y*y) > 1.0) flag = true;
    break;
  case RECTANGLE :    
    if((x*x > m_a*m_a) || (y*y > m_b*m_b)) flag = true;
    break;
  case XFLAT :
    if(x < m_a || x > m_b) flag = true;
    break;
  case YFLAT :
    if(y < m_a || y > m_b) flag = true;//test
    break;    
  default :
    break;
  }

  return flag;

}

void ACCSIM::CollimatorTracker::trackThroughDrift(PAC::Position& pos, double& at) const 
{

  //cout<<"ACCSIM::CollimatorTracker::trackThroughDrift - tracking though drift"<<endl;

  double lstep = getDriftStep(pos, at);
 
  double t0=1.0/sqrt(1.0+(pos[5]+2.0/v0byc)*pos[5]-pos[1]*pos[1]-pos[3]*pos[3]);
  double p1=pos[1]*pos[1]/(t0*t0)+pos[3]*pos[3]/(t0*t0);
  double p4=0.5*(1.0+sqrt(1.0+p1));
  double E=E0+p0*pos[5];
  double cbyrv=E/sqrt(E*E-m0*m0);
  double p2=(1.0/v0byc-cbyrv)*lstep;

  p1=p1/p4*cbyrv*lstep/2.0;
  // std::cout << "Drift, lstep = " << lstep << endl;
  //std::cout << "vbyc = "<<v0byc<<endl;
  // std::cout <<" THe correction factor is "<<t0<<endl;

  if(lstep == 0.0) return;

 
  
  pos[0] += lstep*pos.getPX()*t0; //lstep*vx/vs 
  pos[2] += lstep*pos.getPY()*t0; //lstep*vx/vs 
  //time delay correction
  pos[4] -=p1;
  pos[4] +=p2;

  at += lstep; 
    
}

double ACCSIM::CollimatorTracker::getDriftStep(const PAC::Position& pos, double at) const{

  double step1, step2, step = 0.0;
  double x, px, y, py, a, b, c;

  if(at >= m_length) return step;

  switch(m_shape) {

  case ELLIPSE :

    x  = pos.getX()/m_a;
    px = pos.getPX()/m_a;
    y  = pos.getY()/m_b;
    py = pos.getPY()/m_b;

    a = px*px + py*py;
    b = 2.*(x*px + y*py);
    c = x*x + y*y - 1.0;

    if(a != 0.0) {
      step1 = (-b + sqrt(b*b - 4.0*a*c))/(2.0*a);
      step2 = (-b - sqrt(b*b - 4.0*a*c))/(2.0*a);
    } else {
      step1 = step2 = m_length - at;
    }

    if(step1 >= 0.0) {
      if(step2 >= 0.0){
	step = (step1 < step2) ? step1 : step2;
      }
      else {
	step = step1;
      }
    }
    else {
      if(step2 >= 0.0){
	step = step2;
      }
      else {
	step = 0.0;
      }
    }

    break;

  case RECTANGLE :

    if(pos.getPX() > 0) step1 = (m_a - pos.getX())/pos.getPX();
    else if(pos.getPX() < 0) step1 = (-m_a - pos.getX())/pos.getPX();
    else step1 = m_length - at;

    if(pos.getPY() > 0) step2 = (m_b - pos.getY())/pos.getPY();
    else if(pos.getPY() < 0) step2 = (-m_b - pos.getY())/pos.getPY();
    else step2 = m_length - at;

    step = (step1 < step2) ? step1 : step2;
    break;

  case XFLAT :

    if(pos.getPX() > 0) step = (m_b - pos.getX())/pos.getPX();
    else if(pos.getPX() < 0) step = (m_a - pos.getX())/pos.getPX();
    else step = m_length - at;
    break;

  case YFLAT :
    if(pos.getPY() > 0) step = (m_b - pos.getY())/pos.getPY();
    else if(pos.getPY() < 0) step = (m_a - pos.getY())/pos.getPY();
    else step = m_length - at;
    break;   
 
  default :
    break;

  };

  if(step >= m_length - at) step = m_length - at;
  else {
    // Smooth the target entrance
    step += s_dstep;
  }

  return step;

}

void ACCSIM::CollimatorTracker::trackThroughMaterial(PAC::Particle& part, double& at, double rlam) 
{

  
  double lstep = m_length - at;

  if(std::abs(lstep) < m_radlength*1.e-6) {
    at = m_length;
    return;
  }

  // Get the propagation length

  PAC::Position& pos = part.getPosition();
  double rl = lstep*sqrt(1.0 + pos.getPX()*pos.getPX() + pos.getPY()*pos.getPY());

  //cout<<"ACCSIM::CollimatorTracker::trackThroughMaterial - tracking though material"<<endl;

  // Get the nuclear interaction length

  double rnd = m_uGenerator.getNumber(m_iseed);
  double niL = -rlam*log(rnd);

  // std::cout << "random " << j << " " << rnd << " " << m_iseed << " " 
  // << rlam << " niL = " << niL << " rl = " << rl << endl;  
  //  std::cout << "  Material, lstep = " << lstep << endl;

  // Compare nuclear interaction and propagation lengths

  double mcsL;
  if(niL > rl) mcsL = rl;
  else mcsL = niL;

  // MC scatter
 
  m_scatter.update(part, mcsL, m_iseed);
  at += mcsL/sqrt(1.0 + pos.getPX()*pos.getPX() + pos.getPY()*pos.getPY()); 
  
  m_stopper.update(part, mcsL, m_iseed);

  // Nuclear reaction
  if(niL <= rl) {
    bool apFlag = checkAperture(pos); // added for the ORBIT benchmark
    if(apFlag == 1 && at < m_length) {
      // Nuclear iteraction
      m_nuclearInteraction.update(part, mcsL, m_iseed);
      if(part.getFlag() == 1) {
	m_nLosts++;
	at = m_length;
      }
    }
  }


}

void ACCSIM::CollimatorTracker::init()
{

  // Length
  setLength(0.0);

  // Aperture parameters (shape, a, b)
  setAperture(ACCSIM::CollimatorTracker::ELLIPSE, s_maxsize, s_maxsize);

  // Material parameters (A, Z, rho, radlength)
  setMaterial(1, 1, 1.0, 1.0);

  // Algorithm parameters
  m_iseed = -1;

  m_nLosts = 0;
}

void ACCSIM::CollimatorTracker::copy(const ACCSIM::CollimatorTracker& col)
{

  // Length
  setLength(col.m_length);

  // Aperture parameters (shape, a, b)
  setAperture(col.m_shape, col.m_a, col.m_b);

  // Material parameters
  setMaterial(col.m_A, col.m_Z, col.m_rho, col.m_radlength);

  // Algorithm parameters
  m_iseed    = col.m_iseed;

  m_nLosts   = col.m_nLosts;
}

ACCSIM::CollimatorRegister::CollimatorRegister()
{

  UAL::PropagatorNodePtr nodePtr(new ACCSIM::CollimatorTracker());
  UAL::PropagatorFactory::getInstance().add("ACCSIM::CollimatorTracker", nodePtr);

}

static ACCSIM::CollimatorRegister the_ACCSIM_Register; 
