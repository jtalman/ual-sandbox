//# Library     : UalSXF
//# File        : ElemMltExit.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_EXIT_H
#define UAL_SXF_ELEM_MLT_EXIT_H

#include "ual_sxf/buckets/ElemMultipole.hh"

//
// The ElemMltExit is the SXF adaptor to the SMF collection of
// exit multipole attributes: Multipole bucket (kl, ktl). 
//

class UAL_SXF_ElemMltExit : public UAL_SXF_ElemMultipole
{
public:

  // Constructor.
  UAL_SXF_ElemMltExit(SXF::OStream& out);

  // Return 1.
  int getExitSize() const;

  // Get the SMF exit bucket selected by the given index (0).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getExitBucket(int index);

protected:
  
  // Return exit attributes
  PacElemAttributes* getAttributes(const PacLattElement& element);

};

#endif
