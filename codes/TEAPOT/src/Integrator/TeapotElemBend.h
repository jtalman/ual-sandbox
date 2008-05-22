// Program     : Teapot
// File        : Integrator/TeapotElemBend.h
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#ifndef TEAPOT_ELEM_BEND_H
#define TEAPOT_ELEM_BEND_H

#include "Survey/PacSurveyDrift.h"
#include "Survey/PacSurveySbend.h"

#include "Integrator/TeapotElemSlice.h"

#define TEAPOT_ELEM_BEND_SIZE 2

class TeapotElement;

class TeapotElemBend
{
public:

  // Constructors & copy operator

  TeapotElemBend() { initialize(); }
  TeapotElemBend(const TeapotElemBend& teb) { initialize(teb); }
  ~TeapotElemBend() { erase(); }

  TeapotElemBend& operator=(const TeapotElemBend& teb) { erase(); initialize(teb); return *this; }

  // Access

  double angle() const { return _angle; }
       
  int order() const { return TEAPOT_ELEM_BEND_SIZE - 1; }
  double atw(int index) const { return _atw[index]; }
  double btw(int index) const { return _btw[index]; }

  double ke1() const { return _ke1; }
  double ke2() const { return _ke2; }

  int size() const { return _size; }
  const TeapotElemSlice& slice(int index) const { return _slices[index]; }

  friend class TeapotElement;

protected:

  // Data

  double _angle;
  double _atw[TEAPOT_ELEM_BEND_SIZE];
  double _btw[TEAPOT_ELEM_BEND_SIZE];

  double _ke1;
  double _ke2;

  int _size;
  TeapotElemSlice* _slices;


  // Methods

  void define(double l, double angle, int ir, int element_key);
  void define(PacSurveyData& survey, double l, int ir, int flag); 

  void ke1(double v) { _ke1 = v; }
  void ke2(double v) { _ke2 = v; }

  void propagate(PacSurveyData& survey, double l, int ir) { define(survey, l, ir, 0); }

private:

  // Methods

  void initialize();
  void initialize(const TeapotElemBend& teb);
  void erase();

};

#endif



