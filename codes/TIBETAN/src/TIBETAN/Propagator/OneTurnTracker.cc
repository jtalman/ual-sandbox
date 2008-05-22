
#include "UAL/Common/Def.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "Optics/PacChromData.h"
#include "TIBETAN/Propagator/OneTurnTracker.hh"

// PacLattice TIBETAN::OneTurnTracker::s_lattice;
// Teapot     TIBETAN::OneTurnTracker::s_teapot;

TIBETAN::OneTurnTracker::OneTurnTracker()
{
  init();
}

TIBETAN::OneTurnTracker::OneTurnTracker(const TIBETAN::OneTurnTracker& st)
{
  copy(st);
}

TIBETAN::OneTurnTracker::~OneTurnTracker()
{
}

UAL::PropagatorNode* TIBETAN::OneTurnTracker::clone()
{
  return new TIBETAN::OneTurnTracker(*this);
}

void TIBETAN::OneTurnTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int is0, int is1,
						 const UAL::AttributeSet& attSet)
{

  TIBETAN::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);

  const PacLattice& lattice     = (PacLattice&) sequence;
  PAC::BeamAttributes ba = (PAC::BeamAttributes&) attSet;

  /*
  std::cout << "Initialize lattice s_lattice,name=" << s_lattice.name() 
	    << "lattice.name= " << lattice.name() << std::endl;
  if(s_lattice.name() != lattice.name()){
    std::cout << "use it " << std::endl;
    s_lattice = lattice;
    s_teapot.use(lattice);
  }
  */
  Teapot teapot(lattice);

  // std::cout << "OneTurnTracker: energy = " << ba.getEnergy() << std::endl;

  PacSurveyData surveyData;
  teapot.survey(surveyData);

  m_suml = surveyData.survey().suml();

  // std::cout << "OneTurnTracker: suml = " << m_suml << std::endl;

  PAC::Bunch bunch(1);
  bunch.getBeamAttributes().setRevfreq(UAL::clight/m_suml);

  bunch[0].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0,  1.e-5);

  teapot.track(bunch);

  double ct0 =  bunch[0].getPosition().getCT();
  double de0 =  bunch[0].getPosition().getDE();

  m_alpha0 = (-ct0/UAL::clight)*bunch.getBeamAttributes().getRevfreq()/de0; 

  // std::cout << "OneTurnTracker: m_alpha0 = " << m_alpha0 << std::endl;

  // Transverse part

  PAC::Position orbit;
  teapot.clorbit(orbit, ba);


  PacChromData chr;
  teapot.chrom(chr, ba, orbit);
  
  PacTwissData twiss = chr.twiss();

  m_mux    = twiss.mu(0);
  m_betax  = twiss.beta(0);
  m_alphax = twiss.alpha(0);
  m_dx     = twiss.d(0);
  m_dpx    = twiss.dp(0);

  m_chromx = chr.dmu(0);

  m_muy    = twiss.mu(1);
  m_betay  = twiss.beta(1);
  m_alphay = twiss.alpha(1);
  m_dy     = twiss.d(1);
  m_dpy    = twiss.dp(1);

  m_chromy = chr.dmu(1);

  std::cout << "TIBETAN::OneTurnTracker: " 
	    << " chromx = " << m_chromx/2./UAL::pi 
	    << " chromy = " << m_chromy/2./UAL::pi
	    << std::endl;

 
}

void TIBETAN::OneTurnTracker::propagate(UAL::Probe& probe)
{

  PAC::Bunch& bunch = static_cast<PAC::Bunch& >(probe);

  double revFreq = bunch.getBeamAttributes().getRevfreq();

  // Longitudinal part

  double e       = bunch.getBeamAttributes().getEnergy();
  double m       = bunch.getBeamAttributes().getMass();
  double gamma   = e/m;
  double p2      = e*e - m*m;
  double v0byc2  = p2/(e*e);
  double v0byc   = sqrt(v0byc2);

  double eta0   = m_alpha0 - 1.0/gamma/gamma;
  double a56    = -eta0*m_suml/v0byc;

  // Transverse part

  double gammax  = (1 + m_alphax*m_alphax)/m_betax;
  double gammay  = (1 + m_alphay*m_alphay)/m_betay;

  double mux = m_mux; 
  double muy = m_muy;

  double betax = m_betax;
  double betay = m_betay;
  
  double x, px, y, py, de, ct;
  double xx = 0.0, yy = 0.0;

  for(int ip = 0; ip < bunch.size(); ip++){
    if(bunch[ip].isLost()) {
      std::cout << ip << " particle is lost " << std::endl;
      continue;
    }

    PAC::Position& p = bunch[ip].getPosition();

    x  = p.getX();
    px = p.getPX();
    y  = p.getY();
    py = p.getPY(); 
    de = p.getDE();
    ct = p.getCT();

    double cosx    = cos(mux + m_chromx*de);
    double sinx    = sin(mux + m_chromx*de);

    double cosy    = cos(muy + m_chromy*de);
    double siny    = sin(muy + m_chromy*de);

    p.setX(xx + x*(cosx + m_alphax*sinx) + px*betax*sinx + m_dx*de);
    p.setPX(-x*gammax*sinx + px*(cosx - m_alphax*sinx + m_dpx*de));
    p.setY(yy + y*(cosy + m_alphay*siny) + py*betay*siny + m_dy*de);
    p.setPY(-y*gammay*siny + py*(cosy - m_alphay*siny + m_dpy*de));
    p.setCT(ct + a56*de);

  } 

}

void TIBETAN::OneTurnTracker::init()
{
  // Longitudinal part

  m_suml   = 0.0;
  m_alpha0 = 0.0;

  // Transverse part

  m_mux    = 0.0;
  m_betax  = 0.0;
  m_alphax = 0.0;
  m_dx     = 0.0;
  m_dpx    = 0.0;

  m_chromx = 0.0;

  m_muy    = 0.0;
  m_betay  = 0.0;
  m_alphay = 0.0;
  m_dy     = 0.0;
  m_dpy    = 0.0;

  m_chromy = 0.0;

}

void TIBETAN::OneTurnTracker::copy(const TIBETAN::OneTurnTracker& t)
{
  // Longitudinal part

  m_suml   = t.m_suml;
  m_alpha0 = t.m_alpha0;

  // Transverse part

  m_mux    = t.m_mux;
  m_betax  = t.m_betax;
  m_alphax = t.m_alphax;
  m_dx     = t.m_dx;
  m_dpx    = t.m_dpx;

  m_chromx = t.m_chromx;

  m_muy    = t.m_muy;
  m_betay  = t.m_betay;
  m_alphay = t.m_alphay;
  m_dy     = t.m_dy;
  m_dpy    = t.m_dpy;

  m_chromy = t.m_chromy;

}

TIBETAN::OneTurnTrackerRegister::OneTurnTrackerRegister()
{
  UAL::PropagatorNodePtr nodePtr(new TIBETAN::OneTurnTracker());
  UAL::PropagatorFactory::getInstance().add("TIBETAN::OneTurnTracker", nodePtr);
}

static TIBETAN::OneTurnTrackerRegister theOneTurnTrackerRegister; 


