//# Library     : SXF
//# File        : echo_sxf/EchoSequence.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_SEQUENCE_H
#define SXF_ECHO_SEQUENCE_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /**
   * Implements a sequence echo writer.
   */

  class EchoSequence : public Sequence
  {
  public:

    /** Constructor. */
    EchoSequence(OStream& out); 

    /** Create and return a sequence echo writer. */
    Sequence* clone();

    /** Do nothing, Return true. */
    int openObject(const char* name, const char* type);

    /** Do nothing. */
    void update();

    /** Do nothing. */
    void close();  

    /** Do nothing. */
    void setDesign(const char*);

    /** Do nothing. */
    virtual void setLength(double);

    /** Do nothing. */
    virtual void setAt(double);

    /** Do nothing. */
    virtual void setHAngle(double);

    /** Do nothing. */
    void addNode(AcceleratorNode*);

  };
}

#endif
