//# Library     : UalSXF
//# File        : ElemMltBody.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_BODY_H
#define UAL_SXF_ELEM_MLT_BODY_H

#include "SMF/PacElemBend.h"
#include "ual_sxf/buckets/ElemMultipole.hh"

//
// The ElemMltBody is the SXF adaptor to the SMF collection of
// body multipole attributes: Multipole bucket (kl, ktl). 
//

class UAL_SXF_ElemMltBody : public UAL_SXF_ElemMultipole
{
public:

  // Constructor.
  UAL_SXF_ElemMltBody(SXF::OStream& out);

  // Destructor.
  ~UAL_SXF_ElemMltBody();

  // Zero bucket.
  void close();  

  // Return 2 (Bend and Multipole).
  int getBodySize() const;

  // Get the SMF body bucket selected by the given index (0 or 1).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getBodyBucket(int index);
  
  // Set the value at the current index in the array 
  // attribute (KL or KLS) and increment this index. 
  // Skip the attribute if it is KL[0] and is not a deviation.
  int setArrayValue(double value);

  // Set the value at given index in the array attribute (KL or KLS).
  // Skip the attribute if it is KL[0] and is not a deviation.  
  int setHashValue(double value, int index);

  // Write data
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Return body attributes.
  PacElemAttributes* getAttributes(const PacLattElement& element);

protected:

  // Thin Dipole 

  // Pointer to the SMF bucket.
  PacElemBend* m_pBend;

  // Bend status.
  int m_iBendStatus;

};

#endif
