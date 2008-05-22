
#include "UAL/Common/Def.hh"
#include "Optics/PacChromData.h"
#include "UAL/QT/Player/SectorTracker.hh"


UAL_RHIC::SectorTracker::SectorTracker()
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

  // modulation

  m_dmux60   = 0.001*2.*UAL::pi;
  m_dmuy60   = 0.001*2.*UAL::pi;
}

void UAL_RHIC::SectorTracker::setOptics(UAL::OpticsCalculator& optics)
{
  // Longitudinal part

  m_suml   = optics.suml;
  m_alpha0 = optics.alpha0;

  // Transverse part
  
  PacTwissData twiss = optics.m_chrom->twiss();

  m_mux    = twiss.mu(0);
  m_betax  = twiss.beta(0);
  m_alphax = twiss.alpha(0);
  m_dx     = twiss.d(0);
  m_dpx    = twiss.dp(0);

  m_chromx = optics.m_chrom->dmu(0);

  m_muy    = twiss.mu(1);
  m_betay  = twiss.beta(1);
  m_alphay = twiss.alpha(1);
  m_dy     = twiss.d(1);
  m_dpy    = twiss.dp(1);

  m_chromy = optics.m_chrom->dmu(1);

  // std::cout << "chromx = " <<  m_chromx/UAL::pi/2.0 
  //	    << ", chromy = " << m_chromy/UAL::pi/2.0 
  //	    << std::endl;
  
}

void UAL_RHIC::SectorTracker::propagate(PAC::Bunch& bunch)
{

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
