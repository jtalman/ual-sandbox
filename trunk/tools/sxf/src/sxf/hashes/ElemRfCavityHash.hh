//# Library     : SXF
//# File        : ElemRfCavityHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_RFCAVITY_HASH
#define SXF_ELEM_RFCAVITY_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemRfCavityHash class implements a rf cavity bucket hash
   * (SXF Rf Cavity attribute keys are "volt", "lag", "harmon", 
   * "shunt", and "tfill").
   */

  class ElemRfCavityHash : public ElemBucketHash
  {
  public:

    enum Attributes { VOLT = 0, LAG, HARMON, SHUNT, TFILL, SIZE};

    /** Constructor. */
    ElemRfCavityHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1. 
     */
    virtual int index(const char* name) const;

  };
}

#endif
