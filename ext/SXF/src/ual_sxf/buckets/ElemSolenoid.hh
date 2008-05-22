//# Library     : UalSXF
//# File        : ElemSolenoid.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_SOLENOID_H
#define UAL_SXF_ELEM_SOLENOID_H

#include "SMF/PacElemSolenoid.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemSolenoid is the SXF adaptor to the SMF collection of
// solenoid attributes: Solenoid bucket (ks). The SXF solenoid 
// scalar attributes are ks.
//

class UAL_SXF_ElemSolenoid : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemSolenoid(SXF::OStream& out);

  // Destructor.
  ~UAL_SXF_ElemSolenoid();

  // Zero bucket.
  void close(); 
 
  // Check the scalar attribute key (KS) and set its value. 
  int setScalarValue(double value);

  // Return 1  because the SXF  solenoid attributes are represented by
  // one SMF bucket.
  int getBodySize() const;

  // Get the SMF solenoid bucket selected by the given index (0).
  // Return 0, if the bucket is not defined or empty.
  PacElemBucket* getBodyBucket(int index);

  // Write data.
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Bucket status (defined or empty)
  int m_iSolenoidStatus;

  // Pointer to the Smf bucket
  PacElemSolenoid* m_pSolenoid;

};

#endif
