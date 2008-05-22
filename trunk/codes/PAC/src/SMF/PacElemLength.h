// Library     : PAC
// File        : SMF/PacElemLength.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_LENGTH_H
#define PAC_ELEM_LENGTH_H

#include "SMF/PacElemBucket.h"

class PacElemLength : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemLength() : PacElemBucket(pacLength) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const   { return PAC_LENGTH_SIZE; }

  double l() const { return _data[PAC_LENGTH_L]; }
  double& l()      { return _data[PAC_LENGTH_L]; }

  void l(double v) { _data[PAC_LENGTH_L] = v; }

};

#endif
