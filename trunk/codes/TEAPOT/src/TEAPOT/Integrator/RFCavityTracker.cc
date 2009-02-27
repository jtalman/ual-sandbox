// Library       : TEAPOT
// File          : TEAPOT/Integrator/RFCavityTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 


#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "SMF/PacElemRfCavity.h"
#include "TEAPOT/Integrator/RFCavityTracker.hh"

TEAPOT::RFCavityTracker::RFCavityTracker()
  : TEAPOT::BasicTracker()
{
  init();
}

TEAPOT::RFCavityTracker::RFCavityTracker(const TEAPOT::RFCavityTracker& rft)
  : TEAPOT::BasicTracker(rft)
{
  copy(rft);
}

TEAPOT::RFCavityTracker::~RFCavityTracker()
{
}

void TEAPOT::RFCavityTracker::setRF(double V, double h, double lag)
{
  m_V   = V;
  m_h   = h;
  m_lag = lag;
}

UAL::PropagatorNode* TEAPOT::RFCavityTracker::clone()
{
  return new TEAPOT::RFCavityTracker(*this);
}

void TEAPOT::RFCavityTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						  int is0, 
						  int is1,
						  const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);

   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void TEAPOT::RFCavityTracker::setLatticeElement(const PacLattElement& e)
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

void TEAPOT::RFCavityTracker::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);

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

    // Drift

    p_new = sqrt(e_new*e_new - m0*m0);
    vbyc  = p_new/e_new;

    passDrift(m_l/2., p, v0byc_new, vbyc);
    
  }

  ba.setElapsedTime(t_old + (m_l/v0byc_old + m_l/v0byc_new)/2./UAL::clight);
}

void TEAPOT::RFCavityTracker::init()
{
  m_l   = 0.0;
  m_V   = 0.0;
  m_lag = 0.0;
  m_h   = 0.0;
}

void TEAPOT::RFCavityTracker::copy(const TEAPOT::RFCavityTracker& rft)
{
  m_l   = rft.m_l;
  m_V   = rft.m_V;
  m_lag = rft.m_lag;
  m_h   = rft.m_h;
}

void TEAPOT::RFCavityTracker::passDrift(double l, PAC::Position& p, double v0byc, double vbyc)
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



