//# Library     : SXF
//# File        : ElemCollimatorHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_COLLIMATOR_HASH
#define SXF_ELEM_COLLIMATOR_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemCollimatorHash class implements a collimator bucket hash
   * (SXF Collimator attribute keys are "xsize" and "ysize").
   */

  class ElemCollimatorHash : public ElemBucketHash
  {
  public:

    enum Attributes { XSIZE = 0, YSIZE, SIZE};

    /** Constructor. */
    ElemCollimatorHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1.
     */ 
    virtual int index(const char* name) const;

  };

}

#endif
