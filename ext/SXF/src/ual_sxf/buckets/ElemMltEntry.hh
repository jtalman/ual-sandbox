//# Library     : UalSXF
//# File        : ElemMltEntry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_ENTRY_H
#define UAL_SXF_ELEM_MLT_ENTRY_H

#include "ual_sxf/buckets/ElemMultipole.hh"

//
// The ElemMltEntry is the SXF adaptor to the SMF collection of
// entry multipole attributes: Multipole bucket (kl, ktl). 
//

class UAL_SXF_ElemMltEntry : public UAL_SXF_ElemMultipole
{
public:

  // Constructor.
  UAL_SXF_ElemMltEntry(SXF::OStream& out);

  // Return 1.
  int getEntrySize() const;

  // Get the SMF entry bucket selected by the given index (0).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getEntryBucket(int index);

protected:
  
  // Return entry attributes
  PacElemAttributes* getAttributes(const PacLattElement& element);

};

#endif
