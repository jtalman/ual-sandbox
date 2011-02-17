#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "TEAPOT/Integrator/RFCavityTracker.hh"

#include "SPINK/Propagator/RFSolenoid.hh"

double SPINK::RFSolenoid::m_V = 0;
double SPINK::RFSolenoid::m_h = 0;
double SPINK::RFSolenoid::m_lag = 0;
double SPINK::RFSolenoid::circ = 0.0;

/** pass variables for diagnostics AUL:27APR10 */
bool SPINK::RFSolenoid::coutdmp = 0;
int SPINK::RFSolenoid::nturn = 0;

SPINK::RFSolenoid::RFSolenoid()
  : TEAPOT::BasicTracker()
{
  init();
}

SPINK::RFSolenoid::RFSolenoid(const SPINK::RFSolenoid& rft)
  : TEAPOT::BasicTracker(rft)
{
  copy(rft);
}

SPINK::RFSolenoid::~RFSolenoid()
{
}

/* void SPINK::RFSolenoid::setRF(double V, double h, double lag)*/
//{
//m_V   = V;
//m_h   = h;
//m_lag = lag;
//}

UAL::PropagatorNode* SPINK::RFSolenoid::clone()
{
  return new SPINK::RFSolenoid(*this);
}

void SPINK::RFSolenoid::setLatticeElements(const UAL::AcceleratorNode& sequence,
						  int is0, 
						  int is1,
						  const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);

   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);

}

void SPINK::RFSolenoid::setLatticeElement(const PacLattElement& e)
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
  //cerr << "RFSolenoid" << "\n" ; //AUL:29DEC09
  //cerr << "V = " << m_V << ", lag = " << m_lag << ", harmon = " << m_h << "\n";
}

void SPINK::RFSolenoid::propagate(UAL::Probe& probe)
{
std::cout << "JDT - server side - File " << __FILE__ << " line " << __LINE__ << " __TIMESTAMP__" << __TIMESTAMP__ << " enter method void SPINK::RFSolenoid::propagate(UAL::Probe& probe)\n";
  //   std::cout << "SPINK::RFSolenoid " << m_name << std::endl;

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);

  
  // cerr << "V = " << m_V << ", lag = " << m_lag << ", harmon = " << m_h << ", l = " << m_l << "\n";
  
  // Old beam attributes

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double q           = ba.getCharge();
  double m0          = ba.getMass();
  double e0_old      = ba.getEnergy();
  double p0_old      = sqrt(e0_old*e0_old - m0*m0);
  double v0byc_old   = p0_old/e0_old;
  //double revfreq_old = ba.getRevfreq();
  double t_old       = ba.getElapsedTime();

  // RF attributes
  
  double V   = m_V;
  double lag = m_lag;
  double h   = m_h;

  if( V == 0. ) return; //AUL:20AUG10
  
  /* AUL:17MAR10 ______________ 
  cout << "\nq=" << q << ", m0=" << m0 << ", e0_old=" << e0_old << endl;
  cout << "p0_old=" << p0_old << ", v0byc_old=" << v0byc_old << endl;
  cout << "revfreq_old=" << revfreq_old << ", t_old=" << t_old << endl;
  cout << "circ=" << circ << ", revfrteq_old=" << revfreq_old << endl ;
  */


  // Update the synchronous particle (beam attributes)

  double de0       = q*V*sin(2*UAL::pi*lag);
  double e0_new    = e0_old + de0;
  double p0_new    = sqrt(e0_new*e0_new - m0*m0);
  double v0byc_new = p0_new/e0_new;

  /* AUL:18MAR10 _________ 
  cout << "V=" << V << ", lag=" << lag << endl;
  */

  double revfreq_old = v0byc_old*UAL::clight/circ ; //AUL:18MAR10

  ba.setEnergy(e0_new);
  ba.setRevfreq(revfreq_old*v0byc_new/v0byc_old);

  // Tracking

  PAC::Position tmp;
  double e_old, p_old, e_new, p_new, vbyc, de, phase;

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

    /* * AUL:18MAR10 _______ 
    double gamma = e_old/m0; double gam2 = gamma*gamma; double beta = sqrt(1. - 1./gam2);
    cout << "e0_old=" << e0_old << ", p0_old=" << p0_old << ", DE=" << p.getDE() << ", e_old=" << e_old << endl;
    cout << "p_old=" << p_old << ", vbyc=" << vbyc << ", gamma=" << gamma << ", beta=" << beta << endl;
    */

    passDrift(m_l/2., p, v0byc_old, vbyc);

    // RF

    phase = h*revfreq_old*(p.getCT()/UAL::clight);
    de    = q*V*sin(2.*UAL::pi*(lag - phase)); 

    e_new = e_old + de;
    p.setDE((e_new - e0_new)/p0_new);

    //  AUL:27APR10 ________ 
    if ( coutdmp ){
      std::cout << " " << endl; //AUL:20AUG10
      std::cout << "RFCavityRTracker: "  << m_name << ", turn = " << nturn << endl ;
      std::cout << "V = " << V << ", h = " << h << endl ;
    }


    //  AUL:27APR10 ________ 

    /* AUL:17MAR10
    //std::cout << "h=" << h << "  revfreq_old=" << revfreq_old << std::endl;
    //std::cout << p.getCT() << "  " << p.getDE() << std::endl; //AUL:17MAR10
    std::cout << phase << "  " << de << std::endl; //AUL:17MAR10
    */

    //ßstd::cout << revfreq_old << " " <<phase <<  "  " << e_new << std::endl; //AUL:17MAR10

    // Drift

    p_new = sqrt(e_new*e_new - m0*m0);
    vbyc  = p_new/e_new;

    passDrift(m_l/2., p, v0byc_new, vbyc);
    
  }

  ba.setElapsedTime(t_old + (m_l/v0byc_old + m_l/v0byc_new)/2./UAL::clight);
}

void SPINK::RFSolenoid::init()
{
  m_l   = 0.0;
  m_V   = 0.0;
  m_lag = 0.0;
  m_h   = 0.0;
}

void SPINK::RFSolenoid::copy(const SPINK::RFSolenoid& rft)
{
  m_l   = rft.m_l;
  m_V   = rft.m_V;
  m_lag = rft.m_lag;
  m_h   = rft.m_h;
}

void SPINK::RFSolenoid::passDrift(double l, PAC::Position& p, double v0byc, double vbyc)
{
  // Transverse coordinates

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

}

SPINK::RFSolenoidRegister::RFSolenoidRegister()
{
  UAL::PropagatorNodePtr rfPtr(new SPINK::RFSolenoid());
  UAL::PropagatorFactory::getInstance().add("SPINK::RFSolenoid", rfPtr);
}

static SPINK::RFSolenoidRegister theSpinkRFSolenoidRegister;




