//# Library     : SXF
//# File        : ElemBucketHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_BUCKET_HASH
#define SXF_ELEM_BUCKET_HASH

#include "sxf/Def.hh"

namespace SXF {

  /** 
   * The ElemBucketHash class is an abstact class of all element
   * bucket hashes.
   */

  class ElemBucketHash
  {
  public:

    /** Map an attribute key to its enum value in the bucket. */
    virtual int index(const char* attributeKey) const = 0;

  };
}


#endif
