//# Library     : SXF
//# File        : echo_sxf/EchoParser.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_PARSER_H
#define SXF_ECHO_PARSER_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /** 
   * Implements a user interface to the SXF echo writer.
   */

  class EchoParser
  {
  public:

    /** Constructor. */
    EchoParser();

    /** Parse inputFile and write its echo and detected errors. */
    void read(const char* inputFile, const char* echoFile);

  };
}

#endif
