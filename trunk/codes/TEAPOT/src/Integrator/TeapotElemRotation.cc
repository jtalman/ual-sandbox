// Program     : Teapot
// File        : Integrator/TeapotElemRotation.cc
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#include "Integrator/TeapotElemRotation.h"

void TeapotElemRotation::define(double angle)
{
  _tilt = angle;

  // cos(angle*i)
  for(int i=0; i < _size; i++) _ctilt[i] = cos(angle*(i+1));

  // sin(angle*i)
  for(int j=0; j < _size; j++) _stilt[j] = sin(angle*(j+1));
}

void TeapotElemRotation::initialize()
{
  _tilt = 0.0;

  _size = TEAPOT_ORDER + 1;

  // cos(angle*i)

  _ctilt = new double[_size];

  if(!_ctilt){
    string msg = "Error: TeapotElemRotation::initialize(const TeapotElemRotation& ter) : ";
           msg += "allocation failure \n";
    PacAllocError(msg).raise();
  }

  for(int i=0; i < _size; i++) _ctilt[i] = 1.0;

  // sin(angle*i)

  _stilt = new double[_size];

  if(!_stilt){
    string msg = "Error: TeapotElemRotation::initialize(const TeapotElemRotation& ter) : ";
           msg += "allocation failure \n";
    PacAllocError(msg).raise();
  }

  for(int j=0; j < _size; j++) _stilt[j] = 0.0;

}

void TeapotElemRotation::initialize(const TeapotElemRotation& ter)
{
  _tilt = ter._tilt;

  _size = ter._size;

  // cos(angle*i)

  _ctilt = new double[_size];

  if(!_ctilt){
    string msg = "Error: TeapotElemRotation::initialize(const TeapotElemRotation& ter) : ";
           msg += "allocation failure \n";
    PacAllocError(msg).raise();
  }

  for(int i=0; i < _size; i++) _ctilt[i] = ter._ctilt[i];

  // sin(angle*i)

  _stilt = new double[_size];

  if(!_stilt){
    string msg = "Error: TeapotElemRotation::initialize(const TeapotElemRotation& ter) : ";
           msg += "allocation failure \n";
    PacAllocError(msg).raise();
  }

  for(int j=0; j < _size; j++) _stilt[j] = ter._stilt[j];

}

void TeapotElemRotation::erase()
{
  if(_ctilt) delete [] _ctilt;
  _ctilt = 0;

  if(_stilt) delete [] _stilt;
  _stilt = 0;

  _size = 0;

  _tilt = 0.0;

}
