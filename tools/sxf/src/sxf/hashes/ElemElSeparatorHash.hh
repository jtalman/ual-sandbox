//# Library     : SXF
//# File        : ElemSeparatorHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_ELSEPARATOR_HASH
#define SXF_ELEM_ELSEPARATOR_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemSeparatorHash class implements a separator bucket hash
   * (SXF Separator attribute keys are "el", "ex", and "ey").
   */

  class ElemElSeparatorHash : public ElemBucketHash
  {
  public:

    enum Attributes { EL = 0, EX, EY, SIZE};

    /** Constructor. */
    ElemElSeparatorHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1. 
     */
    virtual int index(const char* name) const;

  };
}

#endif
