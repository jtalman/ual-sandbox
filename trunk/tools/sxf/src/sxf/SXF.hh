//# Library     : SXF
//# File        : SXF.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_H
#define SXF_H

#include "sxf/OStream.hh"
#include "sxf/hashes/ElemAlignHash.hh"
#include "sxf/hashes/ElemApertureHash.hh"
#include "sxf/hashes/ElemBendHash.hh"
#include "sxf/hashes/ElemBeamBeamHash.hh"
#include "sxf/hashes/ElemElSeparatorHash.hh"
#include "sxf/hashes/ElemMultipoleHash.hh"
#include "sxf/hashes/ElemRfCavityHash.hh"
#include "sxf/hashes/ElemSolenoidHash.hh"
#include "sxf/hashes/ElemEmptyHash.hh"
#include "sxf/hashes/ElemCollimatorHash.hh"
#include "sxf/ElemBucketRegistry.hh"
#include "sxf/Element.hh"
#include "sxf/Sequence.hh"
#include "sxf/NodeRegistry.hh"
#include "sxf/AcceleratorReader.hh"

namespace SXF {

  /** Types */
  typedef SXF_Key                 Key;

}


#endif
