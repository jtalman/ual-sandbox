//# Library     : SXF
//# File        : ElemEmptyHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_EMPTY_HASH
#define SXF_ELEM_EMPTY_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemEmptyHash class implements an empty bucket hash.
   */

  class ElemEmptyHash : public ElemBucketHash
  {
  public:

    enum Attributes { SIZE = 0};

    /** Constructor */
    ElemEmptyHash();

    /** Return -1 for any attributes */
    virtual int index(const char* name) const;

  };

}

#endif
