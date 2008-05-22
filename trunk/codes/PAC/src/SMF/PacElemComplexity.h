// Library     : PAC
// File        : SMF/PacElemComplexity.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_COMPLEXITY_H
#define PAC_ELEM_COMPLEXITY_H

#include "SMF/PacElemBucket.h"

class PacElemComplexity : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemComplexity() : PacElemBucket(pacComplexity) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  double n() const { return _data[PAC_COMPLEXITY_N]; }
  double& n()      { return _data[PAC_COMPLEXITY_N]; }

};

#endif
