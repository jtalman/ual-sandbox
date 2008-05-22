// Program     : Teapot
// File        : IntegratorTeapotIntegrator.h
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#include "Integrator/TeapotIntegrator.h"

TeapotIntegrator::TeapotIntegrator()
{
}

int TeapotIntegrator::propagate(const PacGenElement& ge, PAC::BeamAttributes& ba, PAC::Position& p)
{
  return TeapotEngine<double, PAC::Position>::propagate(ge, ba, p);
}

int TeapotIntegrator::propagate(const TeapotElement& te, PAC::BeamAttributes& ba, PAC::Position& p)
{
  return TeapotEngine<double, PAC::Position>::propagate(te, ba, p);
}

int TeapotIntegrator::propagate(const TeapotElement& te, PAC::Position& p, PAC::Position& tmp, 
				PAC::BeamAttributes& ba, double* v0byc)
{
  return TeapotEngine<double, PAC::Position>::propagate(te, p, tmp, ba, v0byc);
}

void TeapotIntegrator::passRfKick(
  int iKick,
  PAC::Position& p,
  PAC::Position& tmp, 
  PAC::BeamAttributes& ba,
  double* v0byc)
{
  // Rf kick

  double rkicks = _kicks[iKick]*_rIr;

  double revfreq_old = ba.getRevfreq();
  double dE = 0, dE0 = 0;
  int order = _rf->order();

  for(int i=0; i <= order; i++){
    dE      += rkicks*_rf->volt(i)*
      sin(2.*PI*(_rf->lag(i) + _rf->harmon(i)*revfreq_old*(p[4]/BEAM_CLIGHT)));
    dE0     += _rf->volt(i)*sin(2.*PI*_rf->lag(i));
  }

  // Recalculate kinematics

  double e0_old    = ba.getEnergy(), m = ba.getMass();
  double p0c_old   = sqrt(e0_old*e0_old -  m*m);
  double v0byc_old = p0c_old/e0_old;

  double e0_new    = e0_old + dE0;
  double p0c_new   = sqrt(e0_new*e0_new -  m*m);
  double v0byc_new = p0c_new/e0_new;

  ba.setEnergy(e0_new);
  ba.setRevfreq(revfreq_old*v0byc_new/v0byc_old);

  // for particle 

  double e_new = p[5]*p0c_old + e0_old + dE;
  p[5] = (e_new - e0_new)/p0c_new;

  // update velocity

  *v0byc = v0byc_new;
  makeVelocity(p, tmp, *v0byc);
  makeRV(ba, p, tmp);

  /*
  for(int j=0; j <= order; j++){
    cerr << j << " rkicks = " << rkicks
	 << " vrf = " << rkicks*_rf->volt(j) 
	 << " freqhnum = " << _rf->harmon(j)
	 << " revfreq = " << ba.revfreq() 
         << " omegabyc = " << _rf->harmon(j)*revfreq_old/BEAM_CLIGHT
         << " lag  = " << _rf->lag(j) << endl;
  }

  double vbyc = 1/tmp[PacPosition::DE];
  double pc_new = sqrt(e_new*e_new - m*m);
  printf("e0 = %15.10e  delta0 = %15.10e v0byc = %15.10e p0 = %15.10e\n", e0_new, dE0, v0byc_new, p0c_new);
  printf("e  = %15.10e  delta  = %15.10e vbyc  = %15.10e p  = %15.10e dp  = %15.10e\n", e_new, dE, vbyc, pc_new,
  (pc_new-p0c_new)/p0c_new);
  */
 
}

int TeapotIntegrator::testAperture(PAC::Position& p)
{ 

  int flag = 0;
  // Aperture

  double x = p[0];
  double y = p[2];

  if(!_aperture) {
    if((x*x + y*y) < TEAPOT_APERTURE) { flag = 0; }
    else                              { flag = 1; }  
    // if(flag) { cerr << "TeapotIntegrator: particle has been lost \n";}
    return flag;
  }

  // Offset

  double xoffset = 0.0;
  double yoffset = 0.0;

  if(_offset){
    xoffset = _offset->dx();
    yoffset = _offset->dy();
  }

  // Size

  double xsize = _aperture->xsize();
  double ysize = _aperture->ysize();

  if(xsize == 0.0) xsize = ysize;
  if(ysize == 0.0) ysize = xsize;

  if(ysize == 0.0) {
    xsize = TEAPOT_APERTURE;
    ysize = TEAPOT_APERTURE;
  }

  x = fabs(x - xoffset);
  y = fabs(y - yoffset);

  // Type

  int aptype = (int) _aperture->shape(); 

  switch (aptype) {
  case 0: // Circle
    if((x*x + y*y) > xsize*xsize) flag = 1;
    break;
  case 1: // Elliptical aperture
    x /= xsize;
    y /= ysize;
    if((x*x + y*y) > 1.0) flag = 1;
    break;
  case 2: // Rectangular aperture
    if((x*x > xsize*xsize) || (y*y > ysize*ysize)) flag = 1;
    break;
  case 3: // Diamond aperture
    x =  xsize*y + ysize*x - xsize*ysize;
    if(x > 0) flag = 1;
    break;
  default:
    break;
  }
  // if(flag) { cerr << "TeapotIntegrator: particle has been lost \n";}
  return flag;
}





