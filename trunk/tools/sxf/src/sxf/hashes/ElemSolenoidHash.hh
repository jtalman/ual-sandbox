//# Library     : SXF
//# File        : ElemSolenoidHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_SOLENOID_HASH
#define SXF_ELEM_SOLENOID_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /**  
   * The ElemSolenoidHash class implements a solenoid bucket hash
   * (SXF Solenoid attribute keys are "ks").
   */

  class ElemSolenoidHash : public ElemBucketHash
  {
  public:

    enum Attributes { KS = 0, SIZE};

    /** Constructor. */
    ElemSolenoidHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1.
     */ 
    virtual int index(const char* name) const;

  };
}

#endif
