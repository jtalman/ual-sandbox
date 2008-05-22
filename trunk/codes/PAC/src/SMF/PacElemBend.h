// Library     : PAC
// File        : SMF/PacElemBend.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_BEND_H
#define PAC_ELEM_BEND_H

#include "SMF/PacElemBucket.h"

class PacElemBend : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemBend() : PacElemBucket(pacBend) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  double angle() const { return _data[PAC_BEND_ANGLE]; }
  double& angle()      { return _data[PAC_BEND_ANGLE]; }

  double fint() const { return _data[PAC_BEND_FINT]; }
  double& fint()      { return _data[PAC_BEND_FINT]; }

};

#endif
