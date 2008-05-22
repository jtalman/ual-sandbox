//# Library     : UAL
//# File        : UAL/ADXF/Writer.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_WRITER_HH
#define UAL_ADXF_WRITER_HH

#include "UAL/ADXF/Def.hh"

namespace UAL {

  /**
   *  The Writer class implements a user interface for writing the SMF containers 
   * into the ADXF file.
   */ 

  class ADXFWriter
  {
  public:

    /** Constructor. */
    ADXFWriter();

    /** Writes data. */
    void write(const char* outFile);

  protected:

    std::string m_tab;

  };

}

#endif
