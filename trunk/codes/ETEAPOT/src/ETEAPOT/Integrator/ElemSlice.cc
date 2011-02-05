// Program     : ETEAPOT
// File        : ETEAPOT/Integrator/ElemSlice.cc
// Copyright   : see Copyright file

#include "ETEAPOT/Integrator/ElemSlice.hh"

ETEAPOT::ElemSlice::ElemSlice()
{
  initialize();
}

ETEAPOT::ElemSlice::ElemSlice(const ETEAPOT::ElemSlice& sl)
{
  initialize(sl);
}

ETEAPOT::ElemSlice& ETEAPOT::ElemSlice::operator=(const ETEAPOT::ElemSlice& sl) 
{
  initialize(sl);
  return *this;
}

PacSurvey& ETEAPOT::ElemSlice::survey() 
{
  return _survey;
}

const PacSurvey& ETEAPOT::ElemSlice::survey() const
{
  return _survey;
}

double& ETEAPOT::ElemSlice::phpl()
{
  return _phpl;
}

double ETEAPOT::ElemSlice::phpl() const
{
  return _phpl;
}

double& ETEAPOT::ElemSlice::cphpl()
{
  return _cphpl;
}

double ETEAPOT::ElemSlice::cphpl() const
{
  return _cphpl;
}

double& ETEAPOT::ElemSlice::sphpl()
{
  return _sphpl;
}

double ETEAPOT::ElemSlice::sphpl() const
{
  return _sphpl;
}

double& ETEAPOT::ElemSlice::tphpl()
{
  return _tphpl;
}

double ETEAPOT::ElemSlice::tphpl() const
{
  return _tphpl;
}

double& ETEAPOT::ElemSlice::rlipl()
{
  return _rlipl;
}

double ETEAPOT::ElemSlice::rlipl() const
{
  return _rlipl;
}

double& ETEAPOT::ElemSlice::scrx()
{
  return _scrx;
}

double ETEAPOT::ElemSlice::scrx() const
{
  return _scrx;
}

double& ETEAPOT::ElemSlice::scrs()
{
  return _scrs;
}

double ETEAPOT::ElemSlice::scrs() const
{
  return _scrs;
}

double& ETEAPOT::ElemSlice::spxt()
{
  return _spxt;
}

double ETEAPOT::ElemSlice::spxt() const
{
  return _spxt;
}

void ETEAPOT::ElemSlice::erase() 
{
  initialize();
}

void ETEAPOT::ElemSlice::initialize()
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

void ETEAPOT::ElemSlice::initialize(const ETEAPOT::ElemSlice& sl)
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

void ETEAPOT::ElemSlice::define(const PacSurvey& previous, const PacSurvey& present, const PacSurvey& next)
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


