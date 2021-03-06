// Library       : TEAPOT
// File          : TEAPOT/Integrator/BasicTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include "TEAPOT/Integrator/BasicTracker.hh"
#include "TEAPOT/Integrator/LostCollector.hh"
#include "SMF/PacLattice.h"
#include <iostream>

using namespace std;
double TEAPOT::BasicTracker::s_maxR = 1.0; 

TEAPOT::BasicTracker::BasicTracker()
{
  initialize();
}

TEAPOT::BasicTracker::BasicTracker(const TEAPOT::BasicTracker& bt)
{
  copy(bt);
}

TEAPOT::BasicTracker::~BasicTracker()
{
}

/*
UAL::PropagatorNode* TEAPOT::BasicTracker::clone()
{
  return new TEAPOT::BasicTracker(*this);
}
*/

void TEAPOT::BasicTracker::initialize()
{
  m_i0 = 0;
  m_i1 = 0;

  m_l  = 0.0;
  m_n  = 0.0;

  m_aperture = 0;
  m_offset   = 0;
}

void TEAPOT::BasicTracker::copy(const TEAPOT::BasicTracker& bt)
{
  m_i0 = bt.m_i0;
  m_i1 = bt.m_i1;

  m_l  = bt.m_l;
  m_n  = bt.m_n;

  m_aperture = bt.m_aperture; 
  m_offset = bt.m_offset; 
}

void TEAPOT::BasicTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);

   m_i0 = is0;
   m_i1 = is1;

   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);

   m_s = lattice[is0].getPosition();
   m_name=lattice[is0].getDesignName();
}

void TEAPOT::BasicTracker::setLatticeElement(const PacLattElement& e)
{
  m_l = e.getLength();
  m_n = e.getN();
  

  // Body attributes
  PacElemAttributes* attributes = e.getBody();
  if(!attributes) return;

  PacElemAttributes::iterator it;

  it = attributes->find(PAC_APERTURE);  
  if(it != attributes->end()) m_aperture = (PacElemAperture*) &(*it);
  
  it = attributes->find(PAC_OFFSET);  
  if(it != attributes->end()) m_offset = (PacElemOffset*) &(*it);

  

}


void TEAPOT::BasicTracker::checkAperture(PAC::Bunch& bunch)
{
  for(int i=0; i < bunch.size(); i++){
    if(bunch[i].isLost()) continue;
    if(!isOK(bunch[i].getPosition())) { 
      bunch[i].setFlag(1); 
      LostCollector::GetInstance().RegisterLoss(i,bunch[i].getPosition(),m_i0,m_s,m_name);
    }
  }
}

bool TEAPOT::BasicTracker::isOK(PAC::Position& p)
{

  bool flag = true;
  // Aperture

  double x = p[0];
  double y = p[2];

  if(!m_aperture) {
    if((x*x + y*y) < s_maxR*s_maxR) { flag = true; }
    else                            { flag = false; }  
    // if(flag) { cerr << "TeapotIntegrator: particle has been lost " << m_i0 << "\n";}
    return flag;
  }

  // Offset

  double xoffset = 0.0;
  double yoffset = 0.0;

  if(m_offset){
    xoffset = m_offset->dx();
    yoffset = m_offset->dy();
  }

  // Size

  double xsize = m_aperture->xsize();
  double ysize = m_aperture->ysize();

  if(xsize == 0.0) xsize = ysize;
  if(ysize == 0.0) ysize = xsize;

  if(ysize == 0.0) {
    xsize = s_maxR;
    ysize = s_maxR;
  }

  x = fabs(x - xoffset);
  y = fabs(y - yoffset);

  // Type

  int aptype = (int) m_aperture->shape(); 

  switch (aptype) {
  case 0: // Circle
    if((x*x + y*y) > xsize*xsize) flag = false;
    break;
  case 1: // Elliptical aperture
    x /= xsize;
    y /= ysize;
    if((x*x + y*y) > 1.0) flag = false;
    break;
  case 2: // Rectangular aperture
    if((x*x > xsize*xsize) || (y*y > ysize*ysize)) flag = false;
    break;
  case 3: // Diamond aperture
    x =  xsize*y + ysize*x - xsize*ysize;
    if(x > 0) flag = false;
    break;
  default:
    break;
  }
  // if(flag) { cerr << "TeapotIntegrator: particle has been lost \n";}
  return flag;
}

