// Library     : Teapot
// File        : Integrator/TeapotElemSlice.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Integrator/TeapotElemSlice.h"

void TeapotElemSlice::initialize(const TeapotElemSlice& sl)
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

void TeapotElemSlice::initialize()
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


void TeapotElemSlice::define(const PacSurvey& previous, const PacSurvey& present, const PacSurvey& next)
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
