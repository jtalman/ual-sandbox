// Library     : PAC
// File        : SMF/PacElemAperture.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_APERTURE_H
#define PAC_ELEM_APERTURE_H

#include "SMF/PacElemBucket.h"

class PacElemAperture : public PacElemBucket
{

public:

  // Constructor & copy operator

  PacElemAperture() : PacElemBucket(pacAperture) {}

  PacElemBucket& operator  = (const PacElemBucket& bucket) { return PacElemBucket::operator=(bucket); }

  // Access

  double shape() const { return _data[PAC_APERTURE_SHAPE]; }
  double& shape()      { return _data[PAC_APERTURE_SHAPE]; }

  double xsize() const { return _data[PAC_APERTURE_XSIZE]; }
  double& xsize()      { return _data[PAC_APERTURE_XSIZE]; }

  double ysize() const { return _data[PAC_APERTURE_YSIZE]; }
  double& ysize()      { return _data[PAC_APERTURE_YSIZE]; }

};

#endif
