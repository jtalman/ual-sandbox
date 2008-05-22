//# Library     : SXF
//# File        : ElemMultipoleHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_MULTIPOLE_HASH
#define SXF_ELEM_MULTIPOLE_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemMultipoleHash class implements a multipole bucket hash
   * (SXF Multipole attribute keys are "kl", "kls", "lrad").
   */

  class ElemMultipoleHash : public ElemBucketHash
  {
  public:

    enum Attributes { KL = 0, KLS, LRAD, SIZE};

    /** Constructor. */
    ElemMultipoleHash();

    /** Map an attribute key to its enum valuem,
     * Return a corresponding enum value or -1. 
     */
    virtual int index(const char* name) const;

  };
}

#endif
