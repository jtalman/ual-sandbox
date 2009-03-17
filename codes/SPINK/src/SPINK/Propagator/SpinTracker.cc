// Library       : SPINK
// File          : SPINK/Propagator/SpinTracker.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/SpinTracker.hh"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

SPINK::SpinTracker::SpinTracker()
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

SPINK::SpinTracker::SpinTracker(const SPINK::SpinTracker& st)
{
  copy(st);
}

SPINK::SpinTracker::~SpinTracker()
{
}

UAL::PropagatorNode* SPINK::SpinTracker::clone()
{
  return new SPINK::SpinTracker(*this);
}


void SPINK::SpinTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

    /*
   std::cout << is0 << " " << lattice[is0].getName() << " " << lattice[is0].getType()  << std::endl;
   if(p_complexity) std::cout << " n = " << p_complexity->n()  << std::endl;
   if(p_length)  std::cout << " l = " << p_length->l() << std::endl;
   if(p_bend)   std::cout <<  " angle = " << p_bend->angle() << std::endl;
   if(p_mlt)    std::cout << " kl1 = "  << p_mlt->kl(1) << std::endl;
   std::cout << std::endl;
   */

}

void SPINK::SpinTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
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

  if(!p_complexity){
    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt

    propagateSpin(b);
    
    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt


    cout.precision(15);
    std::cout << m_name << " x = " << x << ", px = " << px
	      << ", y  = " << y << ", py = " << py
      //	      << ", ct = " << ct<< ", dE = " << de
	      << ", sx = " << sx <<", sy = " << sy
	      << ", sz = " << sz 
	      << endl;


    return;
  }

  int ns = 4*p_complexity->n(); 

  for(int i=0; i < ns; i++) {
    if(p_mlt) *p_mlt /= (2*ns);             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);             // kl, kt

    propagateSpin(b);

    if(p_mlt) *p_mlt /= (2*ns);             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);             // kl, kt

  }


  cout.precision(15);
  std::cout << m_name << " x = " << x << ", px = " << px
	    << ", y  = " << y << ", py = " << py
    //    	    << ", ct = " << ct<< ", dE = " << de
	    << ", sx = " << sx <<", sy = " << sy
	    << ", sz = " << sz 
	    << endl;

}

void SPINK::SpinTracker::setElementData(const PacLattElement& e)
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

void SPINK::SpinTracker::setConventionalTracker(const UAL::AcceleratorNode& sequence,
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

void SPINK::SpinTracker::propagateSpin(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  // Test print Sx by Nikolay March 3 2009
  /*
    double sx = bunch[0].getSpin()->getSX();
    std::cout << m_name << " " << sx << std::endl;
    sx *= 0.9999;
    bunch[0].getSpin()->setSX(sx);
  */
  /*
  std::ofstream out("spin.out");
  char endLine = '\0';
  char line[200];
  */
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
  double beta_s  = ps/es,          gam_s   = es/m0;
  double Ggam_s  = GG*gam_s;
  
  double cc      = 2.99792458E+8;
  
  //double Gm    = 0.0011659230;
  //double Gd    = -0.1429875554;
  //double Gp    = 1.7928456;

  double xw, pxw, yw, pyw, ctw, dew, ct0;
  double Bx = 0.0, By = 0.0, Bz = 0.0, rp_dot_B = 0.0, cof = 0.0; 
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
      Ex   = Er / (1.0 + xw / rho);
      Ey   = Ev;
      Ez   = El;
    }
    
    brho   = 1.0E+9*ps/cc; 
    
    // For a general magnetic field, not including skew quads, solenoid, snake etc.

    Bx     = brho*(k1*yw + 2.0*k2*xw*yw);
    By     = brho*(1.0/rho + k1*xw - k1*yw*yw/2.0/rho + k2*(xw*xw - yw*yw));
    Bz     = 0.0;
     
    vector<double> spin0(3, 0.0), spin(3, 0.0);   
     
    spin0[0] = prt.getSpin()-> getSX();
    spin0[1] = prt.getSpin()-> getSY();
    spin0[2] = prt.getSpin()-> getSZ();

    rp_dot_B = pxw*Bx + pyw*By + Bz*sqrt(1-pxw*pxw-pyw*pyw);
    cof      = sqrt(pxw*pxw + pyw*pyw + ( 1.0 + xw/rho)*(1.0 + xw/rho)) / (1.0E+9 * pw/cc);
    
    
    a1 = cof*((1 + Ggam_w)*Bx - (Ggam_w - GG)*rp_dot_B * pxw 
	       + (Ggam_w + gam_w/(1.0 + gam_w)) * (Ey*sqrt(1.0-pxw*pxw-pyw*pyw)-Ez*pyw)*beta_s/cc)
               + EDM_eta/beta_s/cc*(Ex + cc * beta_s *(pyw*Bz - sqrt(1.0-pxw*pxw-pyw*pyw)*By));
     
    a2 = cof*((1 + Ggam_w)*By - (Ggam_w - GG)*rp_dot_B * pyw
	       + (Ggam_w + gam_w/(1.0 + gam_w))*(Ez*pxw-Ex*sqrt(1.0-pxw*pxw-pyw*pyw))*beta_s/cc)
               + EDM_eta/beta_s/cc*(Ey + cc * beta_s *(sqrt(1.0-pxw*pxw-pyw*pyw)*Bx - pyw*Bz));
     
    a3 = cof*((1 + Ggam_w)*Bz - (Ggam_w - GG)*rp_dot_B * ( 1.0 + xw/rho )
	       + (Ggam_w + gam_w/(1 + gam_w)) * (Ex*pyw - Ey*pxw)*beta_s/cc)
               + EDM_eta/beta_s/cc*(Ez + cc * beta_s *(pxw*By - pyw*Bx)) ;
     
    omega = sqrt( a1*a1 + (a2 - 1.0/rho)*(a2 - 1.0/rho) + a3*a3 );
    mu    = omega * beta_s * (-ctw + ct0 + length/ns/beta_s) * abs(GG)/GG;
    
    ct0 = ctw;

    a1 = a1 / omega;
    a2 = (a2 - 1.0/rho ) / omega;
    a3 = a3 / omega;
    /*
    sprintf(line, "%-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e%c", 
	    Bx,By,Bz,a1,a2,a3,endLine);
    out << line << std::endl;
    */

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

void SPINK::SpinTracker::copy(const SPINK::SpinTracker& st)
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

SPINK::SpinTrackerRegister::SpinTrackerRegister()
{
  UAL::PropagatorNodePtr driftPtr(new SPINK::SpinTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::SpinTracker", driftPtr);
}

static SPINK::SpinTrackerRegister theSpinkSpinTrackerRegister;



