//# Library     : UalSXF
//# File        : ual_sxf/Parser.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_PARSER_H
#define UAL_SXF_PARSER_H

#include "ual_sxf/Def.hh"

//
// The Parser class implements a user interface to the SXF/SMF adaptor.
// 

class UAL_SXF_Parser
{
public:

  // Constructor.
  UAL_SXF_Parser();

  // Read data.
  void read(const char* inputFile, const char* echoFile);

  // Write data.
  void write(const char* outFile);

private:


};

#endif
