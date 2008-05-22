//# Library     : SXF
//# File        : ElemAlignHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_ALIGN_HASH
#define SXF_ELEM_ALIGN_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemAlignHash class implements an align bucket hash
   * (SXF Align attribute keys are "al").
   */

  class ElemAlignHash : public ElemBucketHash
  {
  public:

    enum Attributes { AL = 0, SIZE};

    /** Constructor. */
    ElemAlignHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1. 
     */
    virtual int index(const char* key) const ;

  };

}

#endif
