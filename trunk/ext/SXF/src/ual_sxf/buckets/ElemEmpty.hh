//# Library     : UalSXF
//# File        : ElemEmpty.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_EMPTY_H
#define UAL_SXF_ELEM_EMPTY_H

#include "ual_sxf/ElemBucket.hh"

//
// The ElemEmpty is a reader of empty buckets.
//

class UAL_SXF_ElemEmpty : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemEmpty(SXF::OStream&, const char* type);

  // Ignore unsupported scalar attributes. Return SXF_TRUE.
  int openAttribute(const char* name);

  // Ignore  unsupported scalar attributes. Return SXF_TRUE.
  int setScalarValue(double v);  

  // Ignore unsupported array attributes. Return SXF_TRUE.
  int openArray();

  // Ignore  unsupported array attributes. Return SXF_TRUE.
  int setArrayValue(double v);  
};

#endif
