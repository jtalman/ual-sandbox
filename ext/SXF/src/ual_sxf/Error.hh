//# Library     : SXF
//# File        : ual_sxf/Error.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ERROR
#define UAL_SXF_ERROR

#include "ual_sxf/Element.hh"

// 
// The Error class implements a reader of a wrong element.
//

class UAL_SXF_Error : public UAL_SXF_Element
{
public:
  
  // Constructor.
  UAL_SXF_Error(SXF::OStream& out, const char* type, 
		SXF::ElemBucket* bodyBucket, PacSmf& smf);  

  // Print an error message and return SXF_FALSE
  int openObject(const char* name, const char* type);

};

#endif
