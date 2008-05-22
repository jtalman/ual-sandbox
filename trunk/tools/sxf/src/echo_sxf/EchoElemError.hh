//# Library     : SXF
//# File        : echo_sxf/EchoElemError.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ELEM_ERROR_H
#define SXF_ECHO_ELEM_ERROR_H

#include "echo_sxf/EchoElemBucket.hh"

namespace SXF {

  /** 
   * Implements a echo writer of a wrong
   * element bucket.
   */

  class EchoElemError : public EchoElemBucket
  {
  public:

    /** Constructor. */
    EchoElemError(OStream&);

    /** Print an error message and return false */
    int openObject(const char* name);

  };
}

#endif
