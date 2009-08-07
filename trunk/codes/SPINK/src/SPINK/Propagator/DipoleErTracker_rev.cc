// Library       : SPINK
// File          : SPINK/Propagator/DipoleErTracker.cc
// Copyright     : see Copyright file
// Author        : F.Lin
// C++ version   : N.Malitsky 

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/DipoleErTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

double SPINK::DipoleErTracker::s_er = 0;
double SPINK::DipoleErTracker::s_ev = 0;
double SPINK::DipoleErTracker::s_el = 0;

SPINK::DipoleErTracker::DipoleErTracker()
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

SPINK::DipoleErTracker::DipoleErTracker(const SPINK::DipoleErTracker& st)
{
  copy(st);
}

SPINK::DipoleErTracker::~DipoleErTracker()
{
}

UAL::PropagatorNode* SPINK::DipoleErTracker::clone()
{
  return new SPINK::DipoleErTracker(*this);
}


void SPINK::DipoleErTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

    /*
      std::cout << "DipoleEr  "<<is0 << " " << lattice[is0].getName() << " " << lattice[is0].getType()  << std::endl;
      if(p_complexity) std::cout << " n = " << p_complexity->n()  << std::endl;
      if(p_length)  std::cout << " l = " << p_length->l() << std::endl;
      if(p_bend)   std::cout <<  " angle = " << p_bend->angle() << std::endl;
      if(p_mlt)    std::cout << " kl1 = "  << p_mlt->kl(1) << std::endl;
      std::cout << std::endl;
    */

}

void SPINK::DipoleErTracker::propagate(UAL::Probe& b)
{

  SPINK::SpinTrackerWriter* stw = SPINK::SpinTrackerWriter::getInstance();

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  stw->write(bunch.getBeamAttributes().getElapsedTime());

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

    m_tracker->propagate(bunch);

    t0 += length/v;
    ba.setElapsedTime(t0);

    addErKick(bunch);                    // add electric field

    propagateSpin(b);                    // calculate spin motin using m_bunch2

    return;
  }

  int ns = 4*p_complexity->n();

  length /= ns;

  for(int i=0; i < ns; i++) {

    m_tracker->propagate(bunch);

    t0 += length/v;
    ba.setElapsedTime(t0);

    addErKick(bunch);                     // add electric field

    propagateSpin(b);                     // calculate spin motin using m_bunch2

  }

  /*
  PAC::Position& pos = bunch[0].getPosition();
  
  double sx = bunch[0].getSpin()->getSX();
  double sy = bunch[0].getSpin()->getSY();
  double sz = bunch[0].getSpin()->getSZ();
  double x  = pos.getX();
  double px = pos.getPX();
  double y  = pos.getY();
  double py = pos.getPY();
  double ct = pos.getCT();
  double de = pos.getDE();

  double wp_time = t0 + (-ctw3 / cc);
  
  cout.precision(15);
  std::cout << m_name << " x = " << x << ", px = " << px
	    << ", y  = " << y << ", py = " << py
    //    	    << ", ct = " << ct<< ", dE = " << de
	    << ", sx = " << sx <<", sy = " << sy
	    << ", sz = " << sz 
	    << endl;
  */


}

double SPINK::DipoleErTracker::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);

    return psp0;
}

void SPINK::DipoleErTracker::addErKick(PAC::Bunch& bunch)
{
     // beam data

    PAC::BeamAttributes& ba = bunch.getBeamAttributes();

    double energy = ba.getEnergy();
    double mass   = ba.getMass();
    double charge = ba.getCharge();

    double pc    = sqrt(energy*energy - mass*mass);
    double v0byc = pc/energy;

    //  getting element data

    double length = 0;
    double ang    = 0;

    if(p_length)     length = p_length->l();
    if(p_bend)       ang    = p_bend->angle();

    int ns = 1;
    if(p_complexity) ns = 4*p_complexity->n();

    double h0     = 0; // 1/rho
    if(length) h0 = ang/length;
    
    int size = bunch.size();

    for(int i=0; i < size; i++){

        if(bunch[i].isLost() ) continue;

	PAC::Particle& prt = bunch[i];
	PAC::Position& pos = prt.getPosition();

        double x     = pos.getX();
        double px    = pos.getPX();
    
        double y     = pos.getY();
        double py    = pos.getPY();

	double de    = pos.getDE();
	double ew0   = energy + de;

        //ssh 1 + x/R

        double dxR       = (1. + h0*x);
        double psp0      = get_psp0(pos, v0byc);
        double dxR_by_ps = dxR/psp0;

        // ex, ey, ez

	double ex = s_er/(1. + h0*x)*dxR_by_ps - s_er*dxR;
	//        double ex = s_er*dxR_by_ps - s_er*dxR;
        double ey = s_ev;
	double ez = s_el;

        ex *= charge/v0byc*(length/ns);
        ey *= charge/v0byc*(length/ns);
	ez *= charge/v0byc*(length/ns);

        px += ex;
        py += ey;

	pos.setPX(px);
        pos.setPY(py);

	de  = charge/pc*(s_er/(1.+h0*x)*px + s_ev*py + s_el*(1.+h0*x))*(length/ns);
	pos.setDE(de);

	/*
	cout.precision(15);
	std::cout << " psp0 = " <<  psp0  << endl;
	//	std::cout << " ew = " <<  ew << endl;
	//	std::cout << " de = " <<  de << endl;
	*/
    }
    
}

void SPINK::DipoleErTracker::setElementData(const PacLattElement& e)
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

void SPINK::DipoleErTracker::setConventionalTracker(const UAL::AcceleratorNode& sequence,
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

void SPINK::DipoleErTracker::propagateSpin(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  double length = 0;
  double ang = 0;
  double k1 = 0.0, k2 = 0.0;
  int ns = 0;

  //  getting element data

  if(p_length)     length = p_length->l();
  if(p_bend)       ang    = p_bend->angle();

  if(p_mlt){
    k1   = p_mlt->kl(1)/length;
    k2   = 2.0*p_mlt->kl(2)/length;
  }
  
  if(p_complexity) ns = 4*p_complexity->n();
  if(!p_complexity)ns = 1;   

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double es      = ba.getEnergy(),     m0      = ba.getMass();  
  double GG      = ba.getG(); 
  double EDM_eta = 0.0;  // = ba.getEDMeta();
  double ps      = sqrt(es*es - m0*m0);
  double beta_s  = ps/es,              gam_s   = es/m0;
  double Ggam_s  = GG*gam_s;
  
  double cc      = 2.99792458E+8;
  
  //double Gm    = 0.0011659230;
  //double Gd    = -0.1429875554;
  //double Gp    = 1.7928456;

  double xw, pxw, yw, pyw, ctw, dew, ctw1, ctw3;
  double Bx = 0.0, By = 0.0, Bz = 0.0, rp_dot_B = 0.0, cof = 0.0, v2 = 0.0; 
  double a1 = 0.0, a2 = 0.0, a3 = 0.0;
  double Ex = 0.0, Ey = 0.0, Ez = 0.0, Er = 0.0, Ev = 0.0, El = 0.0;
  double rho=1.0E+12, brho = 0.0, omega = 0.0, mu = 0.0;
  
  //   getting positions and spins
  int size = bunch.size();
   
  for(int i=0; i < size; i++){
    
    if(bunch[i].isLost() ) continue;
    
    PAC::Particle& prt = bunch[i];

    PAC::Position& pos = prt.getPosition();

    xw   = pos.getX(),  pxw  = pos.getPX();
    yw   = pos.getY(),  pyw  = pos.getPY();
    ctw  = pos.getCT(), dew  = pos.getDE();

    double ew     = es + dew*ps;
    double pw     = sqrt(ew*ew - m0*m0);
    double beta_w = pw/ew,    gam_w  = ew/m0;
    double Ggam_w = GG*gam_w;
    
    if(ang){
      rho  = length / ang;
      Ex   = s_er * 1.0E+9 / (1.0 + xw / rho);
      //      Ex   = s_er * 1.0E+9;
      Ey   = s_ev * 1.0E+9;
      Ez   = s_el * 1.0E+9;
    }

    brho   = 1.0E+9*ps/cc; 

    // For a general magnetic field, not including skew quads, solenoid, snake etc.

    Bx     = brho*(k1*yw + 2.0*k2*xw*yw);
    By     = brho*(1.0/rho + k1*xw - k1*yw*yw/2.0/rho + k2*(xw*xw - yw*yw)) 
             + Ex*1.0E+9/(beta_w*cc);
    Bz     = 0.0;
 
    // For proton EDM, only radial electric field in the dipole area

    /*
    Bx     = brho*(k1*yw + 2.0*k2*xw*yw);
    By     = brho*(k1*xw - k1*yw*yw/2.0/rho + k2*(xw*xw - yw*yw)) 
             + Ex*1.0E+9/(beta_w*cc);
    Bz     = 0.0;
    */

    //    std::cout<<brho<<"  "<<rho<<"  "<<" "<<k1<<" "<<ps<<endl;

    vector<double> spin0(3, 0.0), spin(3, 0.0);   
     
    spin0[0] = prt.getSpin()-> getSX();
    spin0[1] = prt.getSpin()-> getSY();
    spin0[2] = prt.getSpin()-> getSZ();

    rp_dot_B = pxw*Bx + pyw*By + Bz*(1.0+xw/rho);
    v2       = pxw*pxw+pyw*pyw+(1.0+xw/rho)*(1.0+xw/rho);

    cof      = sqrt(pxw*pxw + pyw*pyw + ( 1.0 + xw/rho)*(1.0 + xw/rho)) / (1.0E+9 * pw/cc);

    a1 = cof*((1 + Ggam_w)*Bx - (Ggam_w - GG)*rp_dot_B * pxw / v2 
	       + (Ggam_w + gam_w/(1.0 + gam_w)) * (Ey*(1.0+xw/rho)-Ez*pyw)*beta_w/cc)
               + EDM_eta/beta_w/cc*(Ex + cc * beta_w *(pyw*Bz - (1.0+xw/rho)*By));
     
    a2 = cof*((1 + Ggam_w)*By - (Ggam_w - GG)*rp_dot_B * pyw / v2
	       + (Ggam_w + gam_w/(1.0 + gam_w))*(Ez*pxw-Ex*(1.0+xw/rho))*beta_w/cc)
               + EDM_eta/beta_w/cc*(Ey + cc * beta_w *((1.0+xw/rho)*Bx - pyw*Bz));
     
    a3 = cof*((1 + Ggam_w)*Bz - (Ggam_w - GG)*rp_dot_B * (1.0+xw/rho) / v2
	       + (Ggam_w + gam_w/(1 + gam_w)) * (Ex*pyw - Ey*pxw)*beta_w/cc)
               + EDM_eta/beta_w/cc*(Ez + cc * beta_w *(pxw*By - pyw*Bx)) ;
     
    omega = sqrt( a1*a1 + (a2 - 1.0/rho)*(a2 - 1.0/rho) + a3*a3 );
    mu    = omega * length / ns * abs(GG) / GG;
    
    a1 = a1 / omega;
    a2 = (a2 - 1.0/rho ) / omega;
    a3 = a3 / omega;

    double s_mat[3][3];
     
    for(int j=0; j <3; j++){
      for(int k=0; k <3; k++){
	s_mat[j][k] = 0.0;
      }
    }

    double sn  = sin(mu);
    double cs  = 1.0 - cos(mu);
    
    s_mat[0][0] = 1.0 -( a2 * a2 + a3 * a3 ) * cs;
    s_mat[0][1] =         a1 * a2 * cs + a3 * sn;
    s_mat[0][2] =         a1 * a3 * cs - a2 * sn;
    s_mat[1][0] =         a1 * a2 * cs - a3 * sn;
    s_mat[1][1] = 1.0 -( a1 * a1 + a3 * a3 ) * cs;
    s_mat[1][2] =         a2 * a3 * cs + a1 * sn;
    s_mat[2][0] =         a1 * a3 * cs + a2 * sn;
    s_mat[2][1] =         a2 * a3 * cs - a1 * sn;
    s_mat[2][2] = 1.0 -( a1 * a1 + a2 * a2 ) * cs;
    
    for(int j=0; j <3; j++){
      spin[j] = 0.0;
      for(int k=0; k <3; k++){
	spin[j] = spin[j] + s_mat[j][k] * spin0[k];
      }
    }

    prt.getSpin()->setSX(spin[0]);
    prt.getSpin()->setSY(spin[1]);
    prt.getSpin()->setSZ(spin[2]);

    /*    
    for(int j=0; j <3; j++){
      for(int k=0; k <3; k++){
	OTSMat0[j][k] = OTSMat[j][k];
      }
    }
     
    for(int j=0; j <3; j++){
      for(int k=0; k <3; k++){
	OTSMat[j][k] = 0.0;
	for(int l=0; l<3; l++){
	OTSMat[j][k] = OTSMat[j][k] + s_mat[j][l] * OTSMat0[l][k];
	}
      }
    }
    */
    
  }
     
}

void SPINK::DipoleErTracker::copy(const SPINK::DipoleErTracker& st)
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

SPINK::DipoleErTrackerRegister::DipoleErTrackerRegister()
{
  UAL::PropagatorNodePtr dipolePtr(new SPINK::DipoleErTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::DipoleErTracker", dipolePtr);
}

static SPINK::DipoleErTrackerRegister theSpinkDipoleErTrackerRegister;



