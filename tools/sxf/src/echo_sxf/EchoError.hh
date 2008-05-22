//# Library     : SXF
//# File        : echo_sxf/EchoError.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ERROR
#define SXF_ECHO_ERROR

#include "echo_sxf/EchoElement.hh"

namespace SXF {

  /**
   * Implements an echo writer of the wrong element type.
   */

  class EchoError : public EchoElement
  {
  public:
  
    /** Constructor. */
    EchoError(OStream& out, const char* type);  

    /** Print an error message and return false */
    int openObject(const char* name, const char* type);

  };
}

#endif
