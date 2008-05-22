// Library     : PAC
// File        : SMF/PacElemRotation.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_ROTATION_H
#define PAC_ELEM_ROTATION_H

#include "SMF/PacElemBucket.h"

class PacElemRotation : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemRotation() : PacElemBucket(pacRotation) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const { return PAC_ROTATION_SIZE; }

  double dphi() const { return _data[PAC_ROTATION_DPHI]; }
  double& dphi()      { return _data[PAC_ROTATION_DPHI]; }

  double dtheta() const { return _data[PAC_ROTATION_DTHETA]; }
  double& dtheta()      { return _data[PAC_ROTATION_DTHETA]; }

  double tilt() const { return _data[PAC_ROTATION_TILT]; }
  double& tilt()      { return _data[PAC_ROTATION_TILT]; }

};

#endif
