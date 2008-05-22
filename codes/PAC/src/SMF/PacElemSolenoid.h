// Library     : PAC
// File        : SMF/PacElemSolenoid.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_SOLENOID_H
#define PAC_ELEM_SOLENOID_H

#include "SMF/PacElemBucket.h"

class PacElemSolenoid : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemSolenoid() : PacElemBucket(pacSolenoid) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const { return PAC_SOLENOID_SIZE; }

  double ks() const { return _data[PAC_SOLENOID_KS]; }
  double& ks()      { return _data[PAC_SOLENOID_KS]; }

  void ks(double v) { _data[PAC_SOLENOID_KS] = v;    }

};

#endif
