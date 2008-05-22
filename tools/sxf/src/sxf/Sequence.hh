//# Library     : SXF
//# File        : Sequence.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_SEQUENCE_H
#define SXF_SEQUENCE_H

#include "sxf/AcceleratorNode.hh"

namespace SXF {

  /**
   * The Sequence class defines the SXF front end sequence reader 
   * interface. According to the SXF object model, Sequence is a
   * list of accelerator elements and subsequences. 
   */

  class Sequence : public AcceleratorNode
  {
  public:

    /** Return false */
    int isElement() const;

    /** Create a new sequence reader. */
    virtual Sequence* clone() = 0;

    /** Add node. */
    virtual void addNode(AcceleratorNode* node) = 0;

  protected:

    /** Constructor */
    Sequence(OStream& out);  

  };
}

#endif
