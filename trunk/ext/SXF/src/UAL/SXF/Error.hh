//# Library     : UAL
//# File        : UAL/SXF/Error.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ERROR_HH
#define UAL_SXF_ERROR_HH

#include "UAL/SXF/Element.hh"

namespace UAL {

  /** 
   * The Error class implements a reader of a wrong element.
   */

  class SXFError : public SXFElement
  {
  public:
  
    /** Constructor. */
    SXFError(SXF::OStream& out, const char* type, 
	     SXF::ElemBucket* bodyBucket, PacSmf& smf);  

    /** Prints an error message and return SXF_FALSE */
    int openObject(const char* name, const char* type);

  };

}

#endif
