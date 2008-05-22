//# Library     : UalSXF
//# File        : ElemError.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_ERROR_H
#define UAL_SXF_ELEM_ERROR_H

#include "ual_sxf/ElemBucket.hh"

//
// The ElemError implements a reader of wrong buckets.
// 

class UAL_SXF_ElemError : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemError(SXF::OStream&);

  // Print an error message and return SXF_FALSE.
  int openObject(const char* name);

};

#endif
