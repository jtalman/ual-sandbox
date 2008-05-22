// Program     : Teapot
// File        : Integrator/TeapotElemSlice.h
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#ifndef TEAPOT_ELEM_SLICE_H
#define TEAPOT_ELEM_SLICE_H

#include "Survey/PacSurvey.h"

class TeapotElemSlice
{
public:

  // Constructors & copy operator

  TeapotElemSlice() {initialize();}
  TeapotElemSlice(const TeapotElemSlice& sl) { initialize(sl);}

  TeapotElemSlice& operator=(const TeapotElemSlice& sl) { initialize(sl); return *this; }

  // Frame

  PacSurvey& survey() { return _survey; }
  const PacSurvey& survey() const { return _survey; }

  double& phpl()        { return _phpl; }
  double  phpl() const  { return _phpl; }

  double& cphpl()       { return _cphpl; }
  double  cphpl() const { return _cphpl; }

  double& sphpl()       { return _sphpl; }
  double  sphpl() const { return _sphpl; }

  double& tphpl()       { return _tphpl; }
  double  tphpl() const { return _tphpl; }

  double& rlipl()       { return _rlipl; }
  double  rlipl() const { return _rlipl; }

  double& scrx()        { return _scrx; }
  double  scrx() const  { return _scrx; }

  double& scrs()        { return _scrs; }
  double  scrs() const  { return _scrs; }

  double& spxt()        { return _spxt; }
  double  spxt() const  { return _spxt; }

  // Functions

  // define frame parameters
  void define(const PacSurvey& previous, const PacSurvey& present, const PacSurvey& next);

  void erase()  {initialize();}

protected:

  // Data

  PacSurvey _survey;

  // Some precalculations used in tracking

  double _phpl;
  double _cphpl;    // cos(phpl)
  double _sphpl;    // sin(phpl)
  double _tphpl;    // tan(phpl)

  double _rlipl;    // distance between this & next slices
  double _scrx;     // x coordinate of the next slice
  double _scrs;     // s coordinate of the next slice
  double _spxt;     // scrs + scrx*tphpl

protected:

  void initialize();
  void initialize(const TeapotElemSlice& sl);

};

#endif
