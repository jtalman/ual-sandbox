//# Library     : UalSXF
//# File        : ElemRfCavity.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_RFCAVITY_H
#define UAL_SXF_ELEM_RFCAVITY_H

#include "SMF/PacElemRfCavity.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemRfCavity is the SXF adaptor to the SMF collection of
// rf. cavity attributes: RfCavity bucket (volt, lag, harmon). 
// The SXF rf cavity scalar attributes are volt, lag, harmon, shunt,
// and tfill.
//

class UAL_SXF_ElemRfCavity : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemRfCavity(SXF::OStream& out);

  // Destructor.
  ~UAL_SXF_ElemRfCavity();

  // Zero bucket.
  void close();

  // Check the scalar attribute key (VOLT, LAG, HARMON) and set its value. 
  // Skip the SHUNT and TFILL attribute.
  int setScalarValue(double value);

  // Return 1 because the SXF rf attributes are represented by
  // one SMF bucket.
  int getBodySize() const;

  // Get the SMF rf cavity bucket selected by the given index (0).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getBodyBucket(int index);
  
  // Write data.
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Bucket status (defined or empty)
  int m_iRfStatus;

  // Pointer to the Smf bucket.
  PacElemRfCavity* m_pRfCavity;

};

#endif
