// Library       : SPINK
// File          : SPINK/Propagator/DipoleTracker.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky, V.Ptitsyn

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/DipoleTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh" 

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>


SPINK::DipoleTracker::DipoleTracker()
{
  p_entryMlt = 0;
  p_exitMlt = 0;
  p_length = 0;
  p_bend = 0;
  p_mlt = 0;
  p_offset = 0;
  p_rotation = 0;
  // p_aperture = 0;
  p_complexity = 0;
  // p_solenoid = 0;
  // p_rf = 0;
}

/** pass variables for diagnostics AUL:02MAR10 */
bool SPINK::DipoleTracker::coutdmp = 0;
int SPINK::DipoleTracker::nturn = 0;

SPINK::DipoleTracker::DipoleTracker(const SPINK::DipoleTracker& st)
{
  copy(st);
}

SPINK::DipoleTracker::~DipoleTracker()
{
}

UAL::PropagatorNode* SPINK::DipoleTracker::clone()
{
  return new SPINK::DipoleTracker(*this);
}


void SPINK::DipoleTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

}

void SPINK::DipoleTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double energy = ba.getEnergy();
  double mass   = ba.getMass();
  double gam    = energy/mass;

  double p = sqrt(energy*energy - mass*mass);
  double v = p/gam/mass*UAL::clight;

  double t0 = ba.getElapsedTime();

  double length = 0;
  if(p_length)     length = p_length->l();

  if(!p_complexity){

    length /= 2;

    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    propagateSpin(bunch);

    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    return;
  }

  int ns = 4*p_complexity->n();

  length /= 2*ns;

  for(int i=0; i < ns; i++) {

    if(p_mlt) *p_mlt /= (2*ns);          // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);          // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    propagateSpin(bunch);

    if(p_mlt) *p_mlt /= (2*ns);          // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);          // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

  }
}

double SPINK::DipoleTracker::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);

    return psp0;
}


void SPINK::DipoleTracker::setElementData(const PacLattElement& e)
{
 
  // Entry multipole
  PacElemAttributes* front  = e.getFront();
  if(front){
     PacElemAttributes::iterator it = front->find(PAC_MULTIPOLE);
     if(it != front->end()) p_entryMlt = (PacElemMultipole*) &(*it);
  }

  // Exit multipole
  PacElemAttributes* end  = e.getEnd();
  if(end){
     PacElemAttributes::iterator it = end->find(PAC_MULTIPOLE);
     if(it != end->end()) p_exitMlt = (PacElemMultipole*) &(*it);
  }

  // Body attributes
  PacElemAttributes* attributes = e.getBody();

  if(attributes){
    for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++){
      switch((*it).key()){
       case PAC_LENGTH:                          // 1: l
            p_length = (PacElemLength*) &(*it);
            break;
       case PAC_BEND:                            // 2: angle, fint
            p_bend = (PacElemBend*) &(*it);
            break;
       case PAC_MULTIPOLE:                       // 3: kl, ktl
            p_mlt = (PacElemMultipole*) &(*it);
            break;
       case PAC_OFFSET:                          // 4: dx, dy, ds
            p_offset = (PacElemOffset*) &(*it);
            break;
       case PAC_ROTATION:                        // 5: dphi, dtheta, tilt
            p_rotation = (PacElemRotation*) &(*it);
            break;
       case PAC_APERTURE:                        // 6: shape, xsize, ysize
	    // p_aperture = (PacElemAperture*) &(*it);
	    break;
       case PAC_COMPLEXITY:                     // 7: n
            p_complexity = (PacElemComplexity* ) &(*it);
            break;
       case PAC_SOLENOID:                       // 8: ks
            // p_solenoid = (PacElemSolenoid* ) &(*it);
            break;
       case PAC_RFCAVITY:                       // 9: volt, lag, harmon
           // p_rf = (PacElemRfCavity* ) &(*it);
           break;
      default:
	break;
      }
    }
  }

}

void SPINK::DipoleTracker::setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                                int is0, int is1,
                                                const UAL::AttributeSet& attSet)
{
    const PacLattice& lattice = (PacLattice&) sequence;

    double ns = 2;
    if(p_complexity) ns = 8*p_complexity->n();

    UAL::PropagatorNodePtr nodePtr =
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());

    m_tracker = nodePtr;
    
    if(p_complexity) p_complexity->n() = 0;   // ir
    if(p_length)    *p_length /= ns;          // l
    if(p_bend)      *p_bend /= ns;            // angle, fint
     
    m_tracker->setLatticeElements(sequence, is0, is1, attSet);
     
    if(p_bend)      *p_bend *= ns;
    if(p_length)    *p_length *= ns;
    if(p_complexity) p_complexity->n() = ns/8;

}

void SPINK::DipoleTracker::propagateSpin(UAL::Probe& b)
{
    PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
    
    PAC::BeamAttributes& ba = bunch.getBeamAttributes();

    for(int i=0; i < bunch.size(); i++){
        propagateSpin(ba, bunch[i]);
    }
}

void SPINK::DipoleTracker::propagateSpin(PAC::BeamAttributes& ba, PAC::Particle& prt)
{
  // beam attributes

  double e0    = ba.getEnergy();
  double m0    = ba.getMass();
  double GG    = ba.getG();

  double p0    = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;
  // double gamma = e0/m0;

  // position

  PAC::Position& pos = prt.getPosition();

  double x   = pos.getX();
  double px  = pos.getPX();
  double y   = pos.getY();
  double py  = pos.getPY();
  // double ct  = pos.getCT();
  double de  = pos.getDE();

  double pz = get_psp0(pos, v0byc);

  double e = de*p0 + e0;
  double p = sqrt(e*e - m0*m0);

  double gamma = e/m0;

  //  getting element data

  int ns = 1;
  if(p_complexity) ns = 4*p_complexity->n();

  double length = 0.0;
  if(p_length) length = p_length->l()/ns;

  double ang = 0.0, h = 0.0;
  if(p_bend)  {
      ang    = p_bend->angle()/ns;
    // VR added to avoid div by zero Dec22 2010
 if(length > 1E-20)
      h      = ang/length;
  }

  double k1l = 0.0, k2l = 0.0; 
  double k0l = 0.0, kls0 = 0.0; // VR added to handle hkicker and vkicker spin effects Dec22 2010

  if(p_mlt){
    if(p_mlt->order() == 0){
      k0l = p_mlt->kl(0)/ns; kls0 = p_mlt->ktl(0)/ns ;} // VR added to handle hkicker and vkicker spin effects Dec22 2010
    if(p_mlt->order() > 0) k1l   = p_mlt->kl(1)/ns;
    if(p_mlt->order() > 1) k2l   = p_mlt->kl(2)/ns;
  }

  double KLx     = k1l*y + 2.0*k2l*x*y + kls0;
  double KLy     = h*length + k1l*x - k1l*y*y/2.0*h + k2l*(x*x - y*y) + k0l; //VR added kls0 and k0l for kicker field effects.

  if( coutdmp ){ //AUL:02MAR10
    // std::cout << "\nMultipole, turn = " << nturn << endl ;
    // std::cout << m_name << ", KLx = " << KLx << ", KLy = " << KLy  << endl ; //AUL:24FEB10
    // std::cout << "k1l = " << k1l << ", k2l = " << k2l << endl ; //AUL:11MAY10
  }

  double vKL = (px*KLx + py*KLy)/(p/p0);

  double fx = (1.0 + GG*gamma)*KLx - GG*(gamma - 1.0)*vKL*px/(p/p0);
  double fy = (1.0 + GG*gamma)*KLy - GG*(gamma - 1.0)*vKL*py/(p/p0);
  double fz = -GG*(gamma - 1.0)*vKL*pz/(p/p0);

  double dt_by_ds = (1 + h*x)/pz;

  fx *= dt_by_ds;
  fy *= dt_by_ds;
  fz *= dt_by_ds;

  double omega = sqrt(fx*fx + (fy - h*length)*(fy - h*length) + fz*fz);

  if( coutdmp ){std::cout << "omega = " << omega << endl ;} //AUL:01MAR10
  

  //  double a11,a12, a13, a21 ,a22, a23 ,a31, a32, a33;
  // double a1,a2, a3, a4 ,a5, a6 ,a7, a8; // this are named as a(1).. in MAD-SPINK AUL:14DEC09 
  double s_mat[3][3] ;

  if( omega > 0 ) {
    
    double cs = 1.0 - cos(omega); double sn = sin(omega);

    double A[3];
    
    A[0] = fx/omega;
    A[1] = (fy - h*length)/omega;
    A[2] = fz/omega;
    
    if( coutdmp ){  //AUL:02MAR10
      std::cout << "A[0] = " << A[0] << " A[1] = " << A[1] << " A[2] = " <<A[2] << endl ;}

    //AULNLD:11DEC09
    s_mat[0][0] = 1. - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[0][1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[0][2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[1][0] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[1][1] = 1. - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[1][2] =      A[1]*A[2]*cs + A[0]*sn ;
    
    s_mat[2][0] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[2][1] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[2][2] = 1. - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else {
    s_mat[0][0] = s_mat[1][1] = s_mat[2][2] = 1. ;
    s_mat[0][1] = s_mat[0][2] = s_mat[1][0] = s_mat[1][2] = s_mat[2][0] = s_mat[2][1] = 0. ;
  }

  double sx0 = prt.getSpin()-> getSX();
  double sy0 = prt.getSpin()-> getSY();
  double sz0 = prt.getSpin()-> getSZ();

  double sx1 = s_mat[0][0]*sx0 + s_mat[0][1]*sy0 + s_mat[0][2]*sz0;
  double sy1 = s_mat[1][0]*sx0 + s_mat[1][1]*sy0 + s_mat[1][2]*sz0;
  double sz1 = s_mat[2][0]*sx0 + s_mat[2][1]*sy0 + s_mat[2][2]*sz0;
  
  double s2 = sx1*sx1 + sy1*sy1 + sz1*sz1;

  // build One Turn Spin Matrix
  double temp_mat[3][3] ; //dummy matrix


  for(int i=0;i<3; i++)
    {
      for(int k=0;k<3; k++)
	{
	  temp_mat[i][k] = 0. ;
	  for(int j=0;j<3; j++)
	    {
	      temp_mat[i][k] = temp_mat[i][k] + s_mat[i][j]*OTs_mat[j][k] ;
	    }
       	}
    }
  for(int i=0;i<3; i++)
    {
      for(int k=0;k<3; k++)
	{
	  OTs_mat[i][k] = temp_mat[i][k] ;
	}
    }

  if( coutdmp ){ //for diagnostics AUL:02MAR10
    std::cout << "initial Spin =" << sx0 << " " << sy0 << " " << sz0 << "\n";
  std:cout << " final Spin = " << sx1 << " " << sy1 << " " << sz1 << " \n";
    std::cout << "spin matrix" << endl ;
    std::cout << s_mat[0][0] << "  " << s_mat[0][1] << "  " << s_mat[0][2] << endl ;
    std::cout << s_mat[1][0] << "  " << s_mat[1][1] << "  " << s_mat[1][2] << endl ;
    std::cout << s_mat[2][0] << "  " << s_mat[2][1] << "  " << s_mat[2][2] << endl ;
    std::cout << "OT spin matrix" << endl ;
    std::cout << OTs_mat[0][0] << "  " << OTs_mat[0][1] << "  " << OTs_mat[0][2] << endl ;
    std::cout << OTs_mat[1][0] << "  " << OTs_mat[1][1] << "  " << OTs_mat[1][2] << endl ;
    std::cout << OTs_mat[2][0] << "  " << OTs_mat[2][1] << "  " << OTs_mat[2][2] << endl ;
  }

  prt.getSpin()-> setSX(sx1);
  prt.getSpin()-> setSY(sy1);
  prt.getSpin()-> setSZ(sz1);

}

void SPINK::DipoleTracker::copy(const SPINK::DipoleTracker& st)
{
    m_name       = st.m_name;

    p_entryMlt   = st.p_entryMlt;
    p_exitMlt    = st.p_exitMlt;

    p_length     = st.p_length;
    p_bend       = st.p_bend;
    p_mlt        = st.p_mlt;
    p_offset     = st.p_offset;
    p_rotation   = st.p_rotation;
    // p_aperture = st.p_aperture;
    p_complexity = st.p_complexity;
    // p_solenoid = st.p_solenoid;
    // p_rf = st.p_rf;
}

SPINK::DipoleTrackerRegister::DipoleTrackerRegister()
{
  UAL::PropagatorNodePtr dipolePtr(new SPINK::DipoleTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::DipoleTracker", dipolePtr);
}

static SPINK::DipoleTrackerRegister theSpinkDipoleTrackerRegister;
