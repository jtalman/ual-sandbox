//# Library     : SXF
//# File        : echo_sxf/EchoElement.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ELEMENT_H
#define SXF_ECHO_ELEMENT_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /** 
   * Implements an element echo writer.
   */

  class EchoElement : public Element
  {
  public:

    /** Constructor. */
    EchoElement(OStream& out, const char* type, ElemBucket* bodyBucket); 

    /** Destructor. */
    ~EchoElement();

    /** Do nothing */
    int openObject(const char* name, const char* type);

    /** Do nothing */
    void update();

    /** Do nothing */
    void close();  

    /** Do nothing */
    void addBucket(ElemBucket*);

    /** Do nothing */
    void setDesign(const char*);

    /** Do nothing. */
    void setLength(double);

    /** Do nothing. */
    void setAt(double);

    /** Do nothing. */
    void setHAngle(double);

    /** Do nothing.*/
    void setN(double);

  };

}

#endif
