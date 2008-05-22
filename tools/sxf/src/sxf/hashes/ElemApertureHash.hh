//# Library     : SXF
//# File        : ElemApertureHash.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_APERTURE_HASH
#define SXF_ELEM_APERTURE_HASH

#include "sxf/hashes/ElemBucketHash.hh"
 
namespace SXF {

  /** 
   * The ElemApertureHash class implements an aperture bucket hash
   * (SXF Aperture attribute keys are "x", "y", and "shape").

     ammended x ->xsize and y->ysize:  Ray Fliller III
   */

  class ElemApertureHash : public ElemBucketHash
  {
  public:

    enum Attributes { XSIZE = 0, YSIZE, SHAPE, SIZE};

    /** Constructor. */
    ElemApertureHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1.
     */ 
    virtual int index(const char* key) const;

  };

}

#endif
 
