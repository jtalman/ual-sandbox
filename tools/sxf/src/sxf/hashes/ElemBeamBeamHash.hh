//# Library     : SXF
//# File        : ElemBeamBeam.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_BEAMBEAM_HASH
#define SXF_ELEM_BEAMBEAM_HASH

#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemBeamBeamHash class implements a beam-beam bucket hash
   * (SXF beam-beam attribute keys are "sigx", "sigy", "xma", "yma",
   * "npart", and "charge").
   */

  class ElemBeamBeamHash : public ElemBucketHash
  {
  public:

    enum Attributes { SIGX = 0, SIGY, XMA, YMA, NPART, CHARGE, SIZE};

    /** Constructor; */
    ElemBeamBeamHash();

    /** Map an attribute key to its enum value,
     * Return a corresponding enum value or -1.
     */ 
    virtual int index(const char* name) const;

  };

}

#endif
