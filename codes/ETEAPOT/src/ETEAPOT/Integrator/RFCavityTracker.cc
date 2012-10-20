// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/RFCavityTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <cstdlib>

#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "ETEAPOT/Integrator/RFCavityTracker.hh"
#include "ETEAPOT/Integrator/DipoleTracker.hh"

int ETEAPOT::RFCavityTracker::RF=0;
std::string ETEAPOT::RFCavityTracker::RF_m_elementName[1000];
double ETEAPOT::RFCavityTracker::RF_m_sX[1000];

ETEAPOT::RFCavityTracker::RFCavityTracker()
  : ETEAPOT::BasicTracker()
{
  init();
}

ETEAPOT::RFCavityTracker::RFCavityTracker(const ETEAPOT::RFCavityTracker& rft)
  : ETEAPOT::BasicTracker(rft)
{
  copy(rft);
}

ETEAPOT::RFCavityTracker::~RFCavityTracker()
{
}

void ETEAPOT::RFCavityTracker::setRF(double V, double h, double lag)
{
  m_V   = V;
  m_h   = h;
  m_lag = lag;
}

UAL::PropagatorNode* ETEAPOT::RFCavityTracker::clone()
{
  return new ETEAPOT::RFCavityTracker(*this);
}

void ETEAPOT::RFCavityTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						  int is0, 
						  int is1,
						  const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);

   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void ETEAPOT::RFCavityTracker::setLatticeElement(const PacLattElement& e)
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
  // cerr << "V = " << m_V << " lag = " << m_lag << " harmon = " << m_h << "\n";
}

void ETEAPOT::RFCavityTracker::propagate(UAL::Probe& probe)
{
std::cerr << "File " << __FILE__ << " line " << __LINE__ << " enter method void ETEAPOT::RFCavityTracker::propagate(UAL::Probe& probe)\n";
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);

char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

  // cerr << "V = " << m_V << ", lag = " << m_lag << ", harmon = " << m_h << ", l = " << m_l << "\n";
  
  // Old beam attributes

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double q           = ba.getCharge();
  double m0          = ba.getMass();
  double e0_old      = ba.getEnergy();
  double p0_old      = sqrt(e0_old*e0_old - m0*m0);
  double v0byc_old   = p0_old/e0_old;
  double revfreq_old = ba.getRevfreq();
  double t_old       = ba.getElapsedTime();

  // RF attributes
  
  double V   = m_V;
  double lag = m_lag;
  double h   = m_h;

  // Update the synchronous particle (beam attributes)

  double de0       = q*V*sin(2*UAL::pi*lag);
  double e0_new    = e0_old + de0;
  double p0_new    = sqrt(e0_new*e0_new - m0*m0);
  double v0byc_new = p0_new/e0_new;

  ba.setEnergy(e0_new);
  ba.setRevfreq(revfreq_old*v0byc_new/v0byc_old);

  // Tracking

  PAC::Position tmp;
  double e_old, p_old, e_new, p_new, vbyc, de, phase;

  for(int ip = 0; ip < bunch.size(); ip++) {

    PAC::Position& p = bunch[ip].getPosition();

    // Drift

    e_old = p.getDE()*p0_old + e0_old;
    p_old = sqrt(e_old*e_old - m0*m0);
    vbyc  = p_old/e_old;

    passDrift(m_l/2., p, v0byc_old, vbyc);

    // RF

    phase = h*revfreq_old*(p.getCT()/UAL::clight);
    de    = q*V*sin(2.*UAL::pi*(lag - phase)); 

    e_new = e_old + de;
    p.setDE((e_new - e0_new)/p0_new);

#ifndef lngTrmTrk
char buffer [3];
sprintf(buffer,"%d",ip);
std::string bip(buffer);
char buffr2 [10];
sprintf(buffr2,"%+5.2f",ETEAPOT::DipoleTracker::m_m);
std::string bp2(buffr2);
std::string sip = "out/TWISS/StndrdPrtcl";
            sip+=bip;
std::cout << "sip.length() " << sip.length() << "\n";
/*
if(sip.length()==22){sip+="_";}
            sip+="_m=";
            sip+=bp2;
*/
fstream filestr;
filestr.open (sip.c_str(), fstream::out | fstream::app);
filestr << setiosflags( ios::showpos    );  
filestr << setiosflags( ios::uppercase  );  
filestr << setiosflags( ios::scientific );
filestr << setfill( ' ' );
filestr << setiosflags( ios::left );
filestr << setprecision(13) ;
  filestr << ETEAPOT::RFCavityTracker::RF_m_elementName[RF] << " " << ETEAPOT::RFCavityTracker::RF_m_sX[RF] << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << "rf__" << setw(5) << RF << " " << S[ip] << "\n";
filestr.close();
#endif

    // Drift

    p_new = sqrt(e_new*e_new - m0*m0);
    vbyc  = p_new/e_new;

    passDrift(m_l/2., p, v0byc_new, vbyc);
    
  }
RF++;

  ba.setElapsedTime(t_old + (m_l/v0byc_old + m_l/v0byc_new)/2./UAL::clight);
}

void ETEAPOT::RFCavityTracker::init()
{
  m_l   = 0.0;
  m_V   = 0.0;
  m_lag = 0.0;
  m_h   = 0.0;
}

void ETEAPOT::RFCavityTracker::copy(const ETEAPOT::RFCavityTracker& rft)
{
  m_l   = rft.m_l;
  m_V   = rft.m_V;
  m_lag = rft.m_lag;
  m_h   = rft.m_h;
}

void ETEAPOT::RFCavityTracker::passDrift(double l, PAC::Position& p, double v0byc, double vbyc)
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
