//# Library     : UAL
//# File        : ElemError.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_ERROR_HH
#define UAL_SXF_ELEM_ERROR_HH

#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /**
   * The ElemError implements a reader of wrong buckets.
   */

  class SXFElemError : public SXFElemBucket
  {
  public:

    /** Constructor.*/
    SXFElemError(SXF::OStream&);

    /** Print an error message and return SXF_FALSE.*/
    int openObject(const char* name);

  };
}

#endif
