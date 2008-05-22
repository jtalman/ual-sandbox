// Program     : Teapot
// File        : Integrator/TeapotElemRotation.h
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#ifndef TEAPOT_ELEM_ROTATION_H
#define TEAPOT_ELEM_ROTATION_H

#include "Main/TeapotDef.h"

class TeapotElemRotation
{
public:

  // Constrictors & copy operator

  TeapotElemRotation() { initialize(); }
  TeapotElemRotation(const TeapotElemRotation& ter) { initialize(ter); }
 ~TeapotElemRotation() { erase(); }

  TeapotElemRotation& operator=(const TeapotElemRotation& ter) {erase(); initialize(ter); return *this; }

  // 

  void define(double angle);

  // Access

  double tilt() const { return _tilt; }

  int size() const { return _size; }

  double  ctilt(int index) const { return _ctilt[index]; }
  double  stilt(int index) const { return _stilt[index]; }
  
protected:

  // Data

  double _tilt;

  int _size;

  double* _ctilt;
  double* _stilt; 

private:

  // Methods

  void initialize();
  void initialize(const TeapotElemRotation& ter); 
  void erase();

};

#endif
