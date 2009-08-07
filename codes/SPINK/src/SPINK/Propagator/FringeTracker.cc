// Library       : SPINK
// File          : SPINK/Propagator/DipoleErTracker.cc
// Copyright     : see Copyright file
// C++ version   : N.Malitsky, F.Lin

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/FringeTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

double SPINK::FringeTracker::s_er = 0;
double SPINK::FringeTracker::s_ev = 0;
double SPINK::FringeTracker::s_el = 0;

SPINK::FringeTracker::FringeTracker()
{
  p_entryMlt = 0;
  p_exitMlt = 0;
  p_length = 0;
  p_bend = 0;
  p_mlt = 0;
  p_offset = 0;
  p_rotation = 0;
  p_complexity = 0;
}

SPINK::FringeTracker::FringeTracker(const SPINK::FringeTracker& st)
{
  copy(st);
}

SPINK::FringeTracker::~FringeTracker()
{
}

UAL::PropagatorNode* SPINK::FringeTracker::clone()
{
  return new SPINK::FringeTracker(*this);
}


void SPINK::FringeTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

}

void SPINK::FringeTracker::propagate(UAL::Probe& b)
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

  int ns = 4*p_complexity->n();

  length /= ns;

  for(int i=0; i < ns; i++) {

    m_tracker->propagate(bunch);

    t0 += length/v;
    ba.setElapsedTime(t0);

    addErKick(bunch,i);                   // add electric field

    propagateSpin(b,i);                     // calculate spin motin using m_bunch2

  }

}

double SPINK::FringeTracker::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);

    return psp0;
}

void SPINK::FringeTracker::addErKick(PAC::Bunch& bunch, int ii)
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
    double h0     = 0; // 1/rho

    if(p_length)     length = p_length->l();

    int ns = 1;
    if(p_complexity) ns = 4*p_complexity->n();

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

        // ex,ey,ez

	leng   = length / ns * ii;
	sigm   = 0.02;
	sigm2  = sigma**2;
	sigm4  = sigma2**2;
	sigm6  = sigma*sigma4;

	if(m_name="inFr"){
	  double ext = s_er*exp(-(0.05-leng)**2/2.0/sigm2)*[1.0 - x**2/2.0/sigm2 
		       + x**4/8/sigm4 + x**2*(0.05-leng)**2/2/sigm4 
                       - x**4*(0.05-leng)**2/8/sigm6];
	  double eyt = s_ev;
	  double ezt = s_er*x*(0.05-leng)/sigm2*exp(-(0.05-leng)**2/2.0/sigm2)*[-1.0 
		       + x**2/2.0/sigm2 - x**2*(0.05-leng)**2/6/sigm4 ];
	}elseif(m_name="ouFr"){
	  double ext = s_er*exp(-leng**2/2.0/sigm2)*[1.0 - x**2/2.0/sigm2 
                      + x**4/8/sigm4 + x**2*leng**2/2/sigm4 - x**4*leng**2/8/sigm6];
	  double eyt = s_ev;
	  double ezt = s_er*x*leng/sigm2*exp(-leng**2/2.0/sigm2)*[-1.0 
                      + x**2/2.0/sigm2 - x**2*leng**2/6/sigm4 ];
	}else{
	  double ext = 0.0;
	  double eyt = 0.0;
	  double ezt = 0.0;
	}

        ex = ext*charge/v0byc*(length/ns);
        ey = eyt*charge/v0byc*(length/ns);
	ez = ezt*charge/v0byc*(length/ns);

        px += ex;
        py += ey;

	pos.setPX(px);
        pos.setPY(py);

	de  = charge/pc*(ext*px+eyt*py+eyz*(1.+h0*x))*(length/ns);
	pos.setDE(de);

    }
    
}

void SPINK::FringeTracker::setElementData(const PacLattElement& e)
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

void SPINK::FringeTracker::setConventionalTracker(const UAL::AcceleratorNode& sequence,
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

void SPINK::FringeTracker::propagateSpin(UAL::Probe& b, int ii)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  double length = 0;
  double k1 = 0.0, k2 = 0.0;
  int ns = 0;

  //  getting element data

  if(p_length)     length = p_length->l();

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
    
    // ex,ey,ez

    leng   = length / ns * ii;
    sigm   = 0.02;
    sigm2  = sigma**2;
    sigm4  = sigma2**2;
    sigm6  = sigma*sigma4;

    if(m_name='inFr'){
      double Ex = 1.0E+9*s_er*exp(-(0.05-leng)**2/2.0/sigm2)*[1.0 - x**2/2.0/sigm2 
		   + x**4/8/sigm4 + x**2*(0.05-leng)**2/2/sigm4 
		   - x**4*(0.05-leng)**2/8/sigm6];
      double Ey = 1.0E+9*s_ev;
      double Ez = 1.0E+9*s_er*x*(0.05-leng)/sigm2*exp(-(0.05-leng)**2/2.0/sigm2)*[-1.0 
		   + x**2/2.0/sigm2 - x**2*(0.05-leng)**2/6/sigm4 ];
    }elseif(m_name='ouFr'){
      double Ex = 1.0E+9*s_er*exp(-leng**2/2.0/sigm2)*[1.0 - x**2/2.0/sigm2 
		   + x**4/8/sigm4 + x**2*leng**2/2/sigm4 - x**4*leng**2/8/sigm6];
      double Ey = 1.0E+9*s_ev;
      double Ez = 1.0E+9*s_er*x*leng/sigm2*exp(-leng**2/2.0/sigm2)*[-1.0 
		   + x**2/2.0/sigm2 - x**2*leng**2/6/sigm4 ];
    }else{
      double Ex = 0.0;
      double Ey = 0.0;
      double Ez = 0.0;
	}

    brho   = 1.0E+9*ps/cc; 

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

  }
     
}

void SPINK::FringeTracker::copy(const SPINK::FringeTracker& st)
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

SPINK::FringeTrackerRegister::FringeTrackerRegister()
{
  UAL::PropagatorNodePtr dipolePtr(new SPINK::FringeTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::FringeTracker", dipolePtr);
}

static SPINK::FringeTrackerRegister theSpinkFringeTrackerRegister;



