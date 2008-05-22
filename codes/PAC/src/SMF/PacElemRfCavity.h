// Library     : PAC
// File        : SMF/PacElemRfCavity.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_RFCAVITY_H
#define PAC_ELEM_RFCAVITY_H

#include "SMF/PacElemBucket.h"

class PacElemRfCavity : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemRfCavity(int order=0) : PacElemBucket(pacRfCavity, order) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const { return PAC_RFCAVITY_SIZE; }

  double volt(int order) const { return _data[PAC_RFCAVITY_VOLT + order*PAC_RFCAVITY_SIZE]; }
  double& volt(int order)      { return _data[PAC_RFCAVITY_VOLT + order*PAC_RFCAVITY_SIZE]; }

  double lag(int order) const { return _data[PAC_RFCAVITY_LAG + order*PAC_RFCAVITY_SIZE]; }
  double& lag(int order)      { return _data[PAC_RFCAVITY_LAG + order*PAC_RFCAVITY_SIZE]; }

  double harmon(int order) const { return _data[PAC_RFCAVITY_HARMON + order*PAC_RFCAVITY_SIZE]; }
  double& harmon(int order)      { return _data[PAC_RFCAVITY_HARMON + order*PAC_RFCAVITY_SIZE]; }
};

#endif
