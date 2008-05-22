// Library     : PAC
// File        : SMF/PacElemOffset.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_OFFSET_H
#define PAC_ELEM_OFFSET_H

#include "SMF/PacElemBucket.h"

class PacElemOffset : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemOffset() : PacElemBucket(pacOffset) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const { return PAC_OFFSET_SIZE; }

  double dx() const { return _data[PAC_OFFSET_DX]; }
  double& dx()      { return _data[PAC_OFFSET_DX]; }

  double dy() const { return _data[PAC_OFFSET_DY]; }
  double& dy()      { return _data[PAC_OFFSET_DY]; }

  double ds() const { return _data[PAC_OFFSET_DS]; }
  double& ds()      { return _data[PAC_OFFSET_DS]; }

};

#endif
