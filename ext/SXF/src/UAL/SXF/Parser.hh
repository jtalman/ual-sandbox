//# Library     : UAL
//# File        : UAL/SXF/Parser.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_PARSER_HH
#define UAL_SXF_PARSER_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /**
   *  The Parser class implements a user interface to the SXF/SMF adaptor.
   */ 

  class SXFParser
  {
  public:

    /** Constructor. */
    SXFParser();

    /** Reads data. */
    void read(const char* inputFile, const char* echoFile);

    /** Writes data. */
    void write(const char* outFile);

  private:


  };

}

#endif
