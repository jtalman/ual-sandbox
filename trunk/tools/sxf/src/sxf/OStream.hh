//# Library     : SXF
//# File        : OStream.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_OSTREAM_H
#define SXF_OSTREAM_H

#include <iostream>

using namespace std;

namespace SXF {

  /** 
   * The OStream class represents the SXF front end echo writer
   * and an error counter.
   */ 

  class OStream
  {
  public:

    /** Constructor. */
    OStream(ostream& out);

    /** Return an output stream + increment a front end error counter. */
    ostream& cfe_error();

    /** Return an output stream + increment a syntax error counter. */
    ostream& syntax_error();

    /** Set a line number (1). */
    int set_lineno();

    /** Increment a line number. */
    int increment_lineno();

    /** Write a line. */
    void write_line(const char* line);

    /** Write a status. */
    void write_status();

  protected:

    /** Reference to an output stream. */
    ostream& m_refOStream;

    /** Line number. */
    int m_iLineNumber;

    /** Error CFE (front end) and syntax error counters. */
    int m_iCFECounter, m_iSyntaxCounter;

  };

}

#endif
