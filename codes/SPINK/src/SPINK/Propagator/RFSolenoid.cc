#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/RFSolenoid.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

double SPINK::RFSolenoid::RFS_Bdl   = 0;
char   SPINK::RFSolenoid::RFS_rot   = 0;
double SPINK::RFSolenoid::RFS_freq0 = 0;
double SPINK::RFSolenoid::RFS_dfreq = 0;
int    SPINK::RFSolenoid::RFS_nt    = 0;

/** pass variables for diagnostics AUL:02MAR10 */
bool SPINK::RFSolenoid::coutdmp = 0;
int SPINK::RFSolenoid::nturn = 0;

SPINK::RFSolenoid::RFSolenoid()
{
  p_length = 0;
}

SPINK::RFSolenoid::RFSolenoid(const SPINK::RFSolenoid& st)
{
  copy(st);
}

SPINK::RFSolenoid::~RFSolenoid()
{
}

UAL::PropagatorNode* SPINK::RFSolenoid::clone()
{
  return new SPINK::RFSolenoid(*this);
}


void SPINK::RFSolenoid::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

}

void SPINK::RFSolenoid::propagate(UAL::Probe& b)
{
std::cout << "JDT - server side - File " << __FILE__ << " line " << __LINE__ << " __TIMESTAMP__" << __TIMESTAMP__ << " enter method void SPINK::RFSolenoid::propagate(UAL::Probe& b)\n";
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);


  // SPINK::SpinTrackerWriter* stw = SPINK::SpinTrackerWriter::getInstance();
  // stw->write(bunch.getBeamAttributes().getElapsedTime());

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

double SPINK::RFSolenoid::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);

    return psp0;
}


void SPINK::RFSolenoid::setElementData(const PacLattElement& e)
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

void SPINK::RFSolenoid::setConventionalTracker(const UAL::AcceleratorNode& sequence,
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

void SPINK::RFSolenoid::propagateSpin(UAL::Probe& b)
{
    PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
    
    PAC::BeamAttributes& ba = bunch.getBeamAttributes();

    for(int i=0; i < bunch.size(); i++){
        propagateSpin(ba, bunch[i]);
    }
}
/*
void SPINK::RFSolenoid::setSnakeParams(mu1, mu2, phi1, phi2, the1, the2) //AUL:10FEB10
{
  double dtr = 3.1415926536/180.;
  snk1_mu = mu1*dtr  ; snk2_mu = mu2*dtr ;
  snk1_phi = phi1*dtr   ; snk2_phi = phi2*dtr ;
  snk1_theta = the1*dtr    ; snk2_theta = the2*dtr   ;
}
*/

void SPINK::RFSolenoid::propagateSpin(PAC::BeamAttributes& ba, PAC::Particle& prt)
{

   double dtr = atan(1.)/45.;
   /*
   //double snk1_mu    = 180.*dtr  ; double snk2_mu    = 180.*dtr ;
   //double snk1_mu    = 0.*dtr  ; double snk2_mu    = 0.*dtr ;
   //double snk1_phi   = 45.*dtr   ; double snk2_phi   = -45.*dtr ;
   //double snk1_theta = 0.*dtr    ; double snk2_theta = 0.*dtr ;  
  //AUL 10:DEC:09
   */

  double A[3] ;
  double s_mat[3][3] ;

  //snk1_phi *= dtr ; snk1_theta *= dtr ; snk1_mu *= dtr ;
  //snk2_phi *= dtr ; snk2_theta *= dtr ; snk2_mu *= dtr ;

  if( m_name == "snake1") {

    double cs = 1;//1. -cos(snk1_mu*dtr) ; 
    double sn =  0;//sin(snk1_mu*dtr) ;

    A[0] = 1;//cos(snk1_theta*dtr) * sin(snk1_phi*dtr) ; // a(1) in MaD-SPINk
    A[1] = 0;//sin(snk1_theta*dtr) ;                // a(2) in MAD-SPINK
    A[2] = 1;//cos(snk1_theta*dtr) * cos(snk1_phi*dtr) ; // a(3) in MAD-SPINK

    if( coutdmp ){ //AUL:01MAR10
      std::cout << "\nRFSolenoid " << m_name << ", turn = " << nturn << endl ;
//    std::cout << "mu = " << snk1_mu << ", phi = " << snk1_phi << ", theta = " << snk1_theta << endl ;
      std::cout << "A[0] = " << A[0] << ", A[1] = " << A[1] << ", A[2] = " <<A[2] << endl ;
    }

    s_mat[0][0] = 1. - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[0][1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[0][2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[1][0] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[1][1] = 1. - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[1][2] =      A[1]*A[2]*cs + A[0]*sn ;
    
    s_mat[2][0] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[2][1] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[2][2] = 1. - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else if( m_name == "snake2" ) {

    double cs = 1;//1. -cos(snk2_mu*dtr) ;
    double sn = 0;// sin(snk2_mu*dtr) ;

    A[0] = 1;//cos(snk2_theta*dtr) * sin(snk2_phi*dtr) ; // a(1) in MAD-SPINk
    A[1] = 0;//sin(snk2_theta*dtr) ;                // a(2) in MAD-SPINK
    A[2] = 1;//cos(snk2_theta*dtr) * cos(snk2_phi*dtr) ; // a(3) in MAD-SPINK

    if( coutdmp ){ //AUL:01MAR10
      std::cout << "\nRFSolenoid " << m_name << ", turn = " << nturn << endl ;
//    std::cout << "mu = " << snk2_mu << ", phi = " << snk2_phi << ", theta = " << snk2_theta << endl ;
      std::cout << "A[0] = " << A[0] << ", A[1] = " << A[1] << ", A[2] = " <<A[2] << endl ;
    }

    s_mat[0][0] = 1. - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[0][1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[0][2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[1][0] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[1][1] = 1. - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[1][2] =      A[1]*A[2]*cs + A[0]*sn ;
      
    s_mat[2][0] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[2][1] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[2][2] = 1. - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else { //initialize spin matrix at the beginning of a turn
      /*if( nturn == 1 ) //AUL:01MAR10
	{} */
      OTs_mat[0][0] = OTs_mat[1][1] = OTs_mat[2][2] = 1. ;
      OTs_mat[0][1] = OTs_mat[0][2] = OTs_mat[1][0] = OTs_mat[1][2] = OTs_mat[2][0] = OTs_mat[2][1] = 0. ;
      
      if( coutdmp )//AUL:01MAR10
        {
	  std::cout << "\nSpin matrix initialize at " << m_name << ", turn = " << nturn << endl;
	  std::cout << "OT spin matrix" << endl ;
	  std::cout << OTs_mat[0][0] << "  " << OTs_mat[0][1] << "  " << OTs_mat[0][2] << endl ;
	  std::cout << OTs_mat[1][0] << "  " << OTs_mat[1][1] << "  " << OTs_mat[1][2] << endl ;
	  std::cout << OTs_mat[2][0] << "  " << OTs_mat[2][1] << "  " << OTs_mat[2][2] << endl ;
	}
      return ;
  }

  /** propagate spin */
  double sx0 = prt.getSpin()-> getSX();
  double sy0 = prt.getSpin()-> getSY();
  double sz0 = prt.getSpin()-> getSZ();

  double sx1 = s_mat[0][0]*sx0 + s_mat[0][1]*sy0 + s_mat[0][2]*sz0;
  double sy1 = s_mat[1][0]*sx0 + s_mat[1][1]*sy0 + s_mat[1][2]*sz0;
  double sz1 = s_mat[2][0]*sx0 + s_mat[2][1]*sy0 + s_mat[2][2]*sz0;
  
  double s2 = sx1*sx1 + sy1*sy1 + sz1*sz1;

  /** build One Turn spin matrix */
  double temp_mat[3][3] ; //dummy matrix 

  for(int i=0;i<3; i++){
      for(int k=0;k<3; k++){
	  temp_mat[i][k] = 0. ;
	  for(int j=0;j<3; j++){
	    temp_mat[i][k] = temp_mat[i][k] + OTs_mat[i][j]*s_mat[j][k] ;}}}
  for(int i=0;i<3; i++){
      for(int k=0;k<3; k++){
	  OTs_mat[i][k] = temp_mat[i][k] ;  }}

  /** print out matrices */
  if( coutdmp ) //AUL:01MAR10
    { 
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

void SPINK::RFSolenoid::copy(const SPINK::RFSolenoid& st)
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

SPINK::RFSolenoidRegister::RFSolenoidRegister()
{
  UAL::PropagatorNodePtr snakePtr(new SPINK::RFSolenoid());
  UAL::PropagatorFactory::getInstance().add("SPINK::RFSolenoid", snakePtr);
}

static SPINK::RFSolenoidRegister theSpinkRFSolenoidRegister;



