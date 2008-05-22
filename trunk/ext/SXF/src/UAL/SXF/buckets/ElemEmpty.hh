//# Library     : UAL
//# File        : ElemEmpty.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_EMPTY_HH
#define UAL_SXF_ELEM_EMPTY_HH

#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemEmpty is a reader of empty buckets.
   */

  class SXFElemEmpty : public SXFElemBucket
  {
  public:

    /** Constructor.*/
    SXFElemEmpty(SXF::OStream&, const char* type);

    /** Ignore unsupported scalar attributes. Return SXF_TRUE.*/ 
    int openAttribute(const char* name);

    /** Ignore  unsupported scalar attributes. Return SXF_TRUE. */
    int setScalarValue(double v);  

    /** Ignore unsupported array attributes. Return SXF_TRUE. */
    int openArray();

    /** Ignore  unsupported array attributes. Return SXF_TRUE.*/
    int setArrayValue(double v);  
  };

}

#endif
