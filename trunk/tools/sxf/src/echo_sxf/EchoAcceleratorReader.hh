//# Library     : SXF
//# File        : echo_sxf/EchoAcceleratorReader.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ACCELERATOR_READER_H
#define SXF_ECHO_ACCELERATOR_READER_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /** 
   * Implements an interface for coordinating 
   * sequence and element readers/echo writers. 
   */

  class EchoAcceleratorReader : public AcceleratorReader
  {
  public:

    /** Constructor */
    EchoAcceleratorReader(OStream& out);

  };
}

#endif
