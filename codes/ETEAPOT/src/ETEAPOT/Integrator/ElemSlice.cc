// Program     : TEAPOT
// File        : TEAPOT/Integrator/ElemSlice.cc
// Copyright   : see Copyright file
// Author      : L.Schachinger and R.Talman
// C++ version : Nikolay Malitsky

#include "TEAPOT/Integrator/ElemSlice.hh"

TEAPOT::ElemSlice::ElemSlice()
{
  initialize();
}

TEAPOT::ElemSlice::ElemSlice(const TEAPOT::ElemSlice& sl)
{
  initialize(sl);
}

TEAPOT::ElemSlice& TEAPOT::ElemSlice::operator=(const TEAPOT::ElemSlice& sl) 
{
  initialize(sl);
  return *this;
}

PacSurvey& TEAPOT::ElemSlice::survey() 
{
  return _survey;
}

const PacSurvey& TEAPOT::ElemSlice::survey() const
{
  return _survey;
}

double& TEAPOT::ElemSlice::phpl()
{
  return _phpl;
}

double TEAPOT::ElemSlice::phpl() const
{
  return _phpl;
}

double& TEAPOT::ElemSlice::cphpl()
{
  return _cphpl;
}

double TEAPOT::ElemSlice::cphpl() const
{
  return _cphpl;
}

double& TEAPOT::ElemSlice::sphpl()
{
  return _sphpl;
}

double TEAPOT::ElemSlice::sphpl() const
{
  return _sphpl;
}

double& TEAPOT::ElemSlice::tphpl()
{
  return _tphpl;
}

double TEAPOT::ElemSlice::tphpl() const
{
  return _tphpl;
}

double& TEAPOT::ElemSlice::rlipl()
{
  return _rlipl;
}

double TEAPOT::ElemSlice::rlipl() const
{
  return _rlipl;
}

double& TEAPOT::ElemSlice::scrx()
{
  return _scrx;
}

double TEAPOT::ElemSlice::scrx() const
{
  return _scrx;
}

double& TEAPOT::ElemSlice::scrs()
{
  return _scrs;
}

double TEAPOT::ElemSlice::scrs() const
{
  return _scrs;
}

double& TEAPOT::ElemSlice::spxt()
{
  return _spxt;
}

double TEAPOT::ElemSlice::spxt() const
{
  return _spxt;
}

void TEAPOT::ElemSlice::erase() 
{
  initialize();
}

void TEAPOT::ElemSlice::initialize()
{
  // Frame

  _phpl  = 0.;
  _cphpl = 1.;   
  _sphpl = 0.; 
  _tphpl = 0.; 

  _rlipl = 0.;   
  _scrx  = 0.;    
  _scrs  = 0.;  
  _spxt  = 0.;
  
}

void TEAPOT::ElemSlice::initialize(const TEAPOT::ElemSlice& sl)
{
  // Survey

  _survey = sl._survey;

  // Frame

  _phpl  = sl._phpl;
  _cphpl = sl._cphpl;   
  _sphpl = sl._sphpl; 
  _tphpl = sl._tphpl; 

  _rlipl = sl._rlipl;   
  _scrx  = sl._scrx;    
  _scrs  = sl._scrs;  
  _spxt  = sl._spxt;

}

void TEAPOT::ElemSlice::define(const PacSurvey& previous, const PacSurvey& present, const PacSurvey& next)
{
  _survey = present;

  double a_present = 0.5*(previous.theta() + present.theta());  
  double a_next    = 0.5*(present.theta()  + next.theta());

  _phpl = a_next - a_present;
  _cphpl = cos(_phpl);
  _sphpl = sin(_phpl);
  _tphpl = tan(_phpl);

  double zz = next.z() - present.z();
  double xx = next.x() - present.x();
  
  _rlipl = sqrt(zz*zz + xx*xx);
  _scrs  = _rlipl*cos(present.theta() - a_present);
  _scrx  = _rlipl*sin(present.theta() - a_present);
  _spxt  = _scrs + _scrx*_tphpl;

}


