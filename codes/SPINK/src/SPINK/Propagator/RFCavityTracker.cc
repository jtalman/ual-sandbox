#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "TEAPOT/Integrator/RFCavityTracker.hh"

#include "SPINK/Propagator/RFCavityTracker.hh"

double SPINK::RFCavityTracker::m_V = 0;
double SPINK::RFCavityTracker::m_h = 0;
double SPINK::RFCavityTracker::m_lag = 0;
double SPINK::RFCavityTracker::circ = 0.0;

/** pass variables for diagnostics AUL:27APR10 */
bool SPINK::RFCavityTracker::coutdmp = 0;
int SPINK::RFCavityTracker::nturn = 0;

SPINK::RFCavityTracker::RFCavityTracker()
  : TEAPOT::BasicTracker()
{
  init();
}

SPINK::RFCavityTracker::RFCavityTracker(const SPINK::RFCavityTracker& rft)
  : TEAPOT::BasicTracker(rft)
{
  copy(rft);
}

SPINK::RFCavityTracker::~RFCavityTracker()
{
}

/* void SPINK::RFCavityTracker::setRF(double V, double h, double lag)*/
//{
//m_V   = V;
//m_h   = h;
//m_lag = lag;
//}

UAL::PropagatorNode* SPINK::RFCavityTracker::clone()
{
  return new SPINK::RFCavityTracker(*this);
}

void SPINK::RFCavityTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
						  int is0, 
						  int is1,
						  const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);

   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);

}

void SPINK::RFCavityTracker::setLatticeElement(const PacLattElement& e)
{
  init();

  m_l = e.getLength();

  m_V = 0.0;
  m_lag = 0.0;
  m_h = 0.0;


  PacElemAttributes* attributes = e.getBody(); 

  if(attributes == 0) {
    return;
  }

  PacElemAttributes::iterator it = attributes->find(PAC_RFCAVITY);
  if(it != attributes->end()){
    PacElemRfCavity* rfSet = (PacElemRfCavity*) &(*it);
    if(rfSet->order() >= 0){
      m_V = rfSet->volt(0);
      m_lag = rfSet->lag(0);
      m_h = rfSet->harmon(0);
    }
  }
  //cerr << "RFCavityTracker" << "\n" ; //AUL:29DEC09
  //cerr << "V = " << m_V << ", lag = " << m_lag << ", harmon = " << m_h << "\n";
}

void SPINK::RFCavityTracker::propagate(UAL::Probe& probe)
{
  //std::cout << "SPINK::RFCavityTracker " << m_name << std::endl;

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  static int oldturn = -1;
  // if(m_name != "rfac9bnc") return;
  // if(m_name == "rfac9bnc" || m_name == "rfac9mhz" ) {
  if(m_name == "rfac1") {
  // if( nturn == oldturn ) return;
  oldturn = nturn;
  // cerr << "V = " << m_V << ", lag = " << m_lag << ", harmon = " << m_h << ", l = " << m_l << "\n";
  
  // Old beam attributes

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double q           = ba.getCharge();
  double m0          = ba.getMass();
  double e0_old      = ba.getEnergy();
  double p0_old      = sqrt(e0_old*e0_old - m0*m0);
  double v0byc_old   = p0_old/e0_old;
  //  double revfreq_old = ba.getRevfreq();
  double t_old       = ba.getElapsedTime();

  // RF attributes
  
  double V   = m_V;
  double lag = m_lag;
  double h   = m_h;
  
  /* AUL:17MAR10 ______________ 
  cout << "\nq=" << q << ", m0=" << m0 << ", e0_old=" << e0_old << endl;
  cout << "p0_old=" << p0_old << ", v0byc_old=" << v0byc_old << endl;
  cout << "revfreq_old=" << revfreq_old << ", t_old=" << t_old << endl;
  cout << "circ=" << circ << ", revfrteq_old=" << revfreq_old << endl ;
  */

  // lag = 0.0;
  // Update the synchronous particle (beam attributes)
  
  double de0       = q*V*sin(2*UAL::pi*lag);
  double e0_new    = e0_old + de0;
  double p0_new    = sqrt(e0_new*e0_new - m0*m0);
  double v0byc_new = p0_new/e0_new;

  /* AUL:18MAR10 _________ 
  cout << "V=" << V << ", lag=" << lag << endl;
  */

  double revfreq_old = v0byc_old*UAL::clight/circ ; //AUL:18MAR10
  //double revfreq_old = UAL::clight/circ;
  //cout << "setting new energy =" << e0_new << " \n";
  ba.setEnergy(e0_new);
  double revfreq_new = revfreq_old*v0byc_new/v0byc_old;
  ba.setRevfreq(revfreq_old*v0byc_new/v0byc_old);

  // Tracking

  PAC::Position tmp;
  double e_old, p_old, e_new, p_new, vbyc, de, phase,dp;

  for(int ip = 0; ip < bunch.size(); ip++) {

    PAC::Position& p = bunch[ip].getPosition();

    /*    
    std::cout << "RF: x = " << p[0] << std::endl; //AUL:17MAR10
    std::cout << "RF: xp = " << p[1] << std::endl; //AUL:17MAR10
    std::cout << "RF: y = " << p[2] << std::endl; //AUL:17MAR10
    std::cout << "RF: yp = " << p[3] << std::endl; //AUL:17MAR10
    std::cout << "RF: ct = " << p[4] << std::endl; //AUL:17MAR10
    std::cout << "RF: dp = " << p[5] << std::endl; //AUL:17MAR10
    */

    // Drift

    e_old = p.getDE()*p0_old + e0_old;
    p_old = sqrt(e_old*e_old - m0*m0);
    vbyc  = p_old/e_old;

    //    e_old = e_old*v0byc_new/v0byc_old;

    passDrift(m_l/2., p, v0byc_old, vbyc);
    // lag = 0.0;
   
    // RF
    dp = p[5];
    phase = h*revfreq_old*(p.getCT()/UAL::clight);
    de    = q*V*sin(2.*UAL::pi*(lag - phase-0.5)); 
    //  dp +=   q*V*sin(2.*UAL::pi*phase)/e0_old;
    e_new = e_old + de;
  
	   p.setDE((e_new - e0_new)/p0_new);
           p.setPX(p[1]*p0_old/p0_new);
           p.setPY(p[3]*p0_old/p0_new);
    //  p.setDE(dp);
  
  //  AUL:27APR10 ________ 
	   if ( coutdmp ){ 
      std::cout << "RFCavityRTracker: "  << m_name << ", turn = " << nturn << endl ;
      std::cout << "V = " << V << ", h*revfreq_old/c = " << h*revfreq_old/UAL::clight << endl ;
      std::cout << "revfreq_old/revfreq_new =" << revfreq_old/revfreq_new << "\n";    }
  

    // Drift

    p_new = sqrt(e_new*e_new - m0*m0);
    vbyc  = p_new/e_new;

    passDrift(m_l/2., p, v0byc_new, vbyc);
    
  }

  ba.setElapsedTime(t_old + (m_l/v0byc_old + m_l/v0byc_new)/2./UAL::clight);
 }
}

void SPINK::RFCavityTracker::init()
{
  m_l   = 0.0;
  m_V   = 0.0;
  m_lag = 0.0;
  m_h   = 0.0;
}

void SPINK::RFCavityTracker::copy(const SPINK::RFCavityTracker& rft)
{
  m_l   = rft.m_l;
  m_V   = rft.m_V;
  m_lag = rft.m_lag;
  m_h   = rft.m_h;
}

void SPINK::RFCavityTracker::passDrift(double l, PAC::Position& p, double v0byc, double vbyc)
{
  // Transverse coordinates
  //  std::cout << "v0byc =" << v0byc << " vbyc = " << vbyc << " \n";

  double ps2_by_po2 = 1. + (p[5] + 2./v0byc)*p[5] - p[1]*p[1] - p[3]*p[3];
  double t0 = 1./sqrt(ps2_by_po2);

  double px_by_ps = p[1]*t0;
  double py_by_ps = p[3]*t0;

  p[0] += (l*px_by_ps);                
  p[2] += (l*py_by_ps);

  // Longitudinal part

  // ct = L/(v/c) - Lo/(vo/c) = (L - Lo)/(v/c) + Lo*(c/v - c/vo) = 
  //                          = cdt_circ       + cdt_vel

  // 1. cdt_circ = (c/v)(L - Lo) = (c/v)(L**2 - Lo**2)/(L + Lo)

  double dl2_by_lo2  = px_by_ps*px_by_ps + py_by_ps*py_by_ps; // (L**2 - Lo**2)/Lo**2
  double l_by_lo     = sqrt(1. + dl2_by_lo2);                 // L/Lo
  
  double cdt_circ = dl2_by_lo2*l/(1 + l_by_lo)/vbyc;

  // 2. cdt_vel = Lo*(c/v - c/vo)

  double cdt_vel = l*(1./vbyc - 1./v0byc);

  // MAD longitudinal coordinate = -ct 

  p[4] -= cdt_vel + cdt_circ;

  //std::cout << "ct in drift =" << p[4] << " \n";
  //std::cout << "cdt_vel =" << cdt_vel << "cdt_circ =" << cdt_circ << " \n";
}

SPINK::RFCavityTrackerRegister::RFCavityTrackerRegister()
{
  UAL::PropagatorNodePtr rfPtr(new SPINK::RFCavityTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::RFCavityTracker", rfPtr);
}

static SPINK::RFCavityTrackerRegister theSpinkRFCavityTrackerRegister;




