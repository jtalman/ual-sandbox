// Library     : PAC
// File        : SMF/PacElemMultipole.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_MULTIPOLE_H
#define PAC_ELEM_MULTIPOLE_H

#include "SMF/PacElemBucket.h"

class PacElemMultipole : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemMultipole(int order=0) : PacElemBucket(pacMultipole, order) {}
  PacElemMultipole(const PacElemBucket& bucket) : PacElemBucket(bucket) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  int keySize() const { return PAC_MULTIPOLE_SIZE; }

  double kl(int order) const { return _data[PAC_MULTIPOLE_KL + order*PAC_MULTIPOLE_SIZE]; }
  double& kl(int order)      { return _data[PAC_MULTIPOLE_KL + order*PAC_MULTIPOLE_SIZE]; }

  double ktl(int order) const { return _data[PAC_MULTIPOLE_KTL + order*PAC_MULTIPOLE_SIZE]; }
  double& ktl(int order)      { return _data[PAC_MULTIPOLE_KTL + order*PAC_MULTIPOLE_SIZE]; }
};

#endif
