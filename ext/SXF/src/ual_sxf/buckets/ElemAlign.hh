//# Library     : UalSXF
//# File        : ElemAlign.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_ALIGN_H
#define UAL_SXF_ELEM_ALIGN_H

#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemAlign class is the SXF adaptor to the SMF collections
// of align attributes: Offset (dx, dy, ds) and Rotation (dphi, 
// dtheta, tilt) buckets. The SXF align attributes are represented
// by the AL array, [dx, dy, ds,  dphi, dtheta, tilt].
//

class UAL_SXF_ElemAlign : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemAlign(SXF::OStream&);

  // Destructor.
  ~UAL_SXF_ElemAlign();

  // Zero bucket.
  void close();

  // Check the attribute key.
  // Return SXF_TRUE or SXF_FALSE.
  int openArray();

  // Do nothing.
  void closeArray();

  // Do nothing. Return SXF_TRUE.
  int openHash();

  // Do nothing.
  void closeHash();  

  // Set the value at the current index in the array attribute (AL)
  // and increment this index. Return SXF_TRUE or SXF_FALSE.
  int setArrayValue(double value);

  // Set the value at given index in the array attribute (AL).
  // Return SXF_TRUE or SXF_FALSE.
  int setHashValue(double value, int index);

  // Return 2 because the align attributes are represented by two 
  // SMF buckets: Offset and Rotation.
  int getBodySize() const;

  // Get the SMF body bucket selected by the given index (0 or 1).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getBodyBucket(int index);

  // Write data.
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Current index in the multi-valued indexed attribute.
  int m_iIndex;

  // Bucket status (defined or empty).
  int m_iOffsetStatus;
  int m_iRotationStatus;

  // Pointer to the SMF buckets.
  PacElemOffset*   m_pOffset;
  PacElemRotation* m_pRotation;


};

#endif
