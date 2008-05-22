// Library     : PAC
// File        : SMF/PacSmf.h
// Copyright   : see Copyright file
// Description : Standard Machine Format
// Author      : Nikolay Malitsky

#ifndef PAC_SMF_H
#define PAC_SMF_H

#include "SMF/PacElemKeys.h"
#include "SMF/PacElemBucketKeys.h"
#include "SMF/PacGenElements.h"
#include "SMF/PacLines.h"
#include "SMF/PacLattices.h"

class PacSmf
{
public:

  PacSmf();

// Primary collections

  // Extent of design elements
  PacGenElements* elements();

  // Extent of design beam lines 
  PacLines* lines();

  // Extent of real lattices
  PacLattices* lattices();

// Secondary collections

  // Extent of element keys
  PacElemKeys*  elemKeys();

  // Extent of element bucket keys
  PacElemBucketKeys* bucketKeys();

};


#endif
