//# Library     : UalSXF
//# File        : ElemKicker.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_KICKER_H
#define UAL_SXF_ELEM_KICKER_H

#include "SMF/PacElemMultipole.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemKicker is the SXF adaptor to the SMF collection of
// kicker attributes: Multipole (kl, ktl) bucket. The SXF kicker
// attributes are represented by two scalars, kl and kls.
//

class UAL_SXF_ElemKicker : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemKicker(SXF::OStream&);

  // Destructor.
  ~UAL_SXF_ElemKicker();

  // Zero bucket.
  void close();

  // Check the scalar attribute key (KL , KLS) and set its value. 
  // Return SXF_TRUE or SXF_FALSE.
  int setScalarValue(double); 

  // Write data.
  void write(ostream& out, const PacLattElement& element, const string& tab);

  // Return 1 because the SXF kicker attributes are represented
  // by one SMF Multipole bucket.
  int getBodySize() const;

  // Get the SMF body bucket selected by the given index (0).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getBodyBucket(int index);


protected:

  // Bucket status (defined or empty)
  int m_iMltStatus;

  // Pointer to the Smf bucket
  PacElemMultipole* m_pMultipole;

};

#endif
