//# Library     : SXF
//# File        : SequenceStack.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky


#ifndef SXF_SEQUENCE_STACK_H
#define SXF_SEQUENCE_STACK_H

#include "sxf/Sequence.hh"

namespace SXF {

  /** 
   * The SequenceStack implements a stack of nested sequence readers.
   */

  class SequenceStack
  {
  public:

    /** Constructor. */
    SequenceStack(int size);
  
    /** Destructor. */
    ~SequenceStack();

    /** Return the max stack size. */
    int size() const;

    /** Return true if stack is empty. */
    int isEmpty() const;

    /** Return and remove the topmost element in the stack. */
    Sequence* pop();

    /** Return the topmost element in the stack. */
    Sequence* top();

    /** Puch new value on the stack. */
    void push(Sequence* value);

  protected:

    /** Stack size. */
    int m_iSize;

    /** Stack elements. */
    Sequence** m_pSequences;

    /** Stack cursor. */
    int m_iNextSlot;

  };
}


#endif
