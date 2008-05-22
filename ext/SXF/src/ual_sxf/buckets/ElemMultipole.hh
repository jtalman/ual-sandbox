//# Library     : UalSXF
//# File        : ElemMultipole.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MULTIPOLE_H
#define UAL_SXF_ELEM_MULTIPOLE_H

#include "SMF/PacElemMultipole.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemMultipole class defines a basis interface and
// a common behavior of the SMF adaptors to the SMF 
// collections of the multipole attributes.
//

class UAL_SXF_ElemMultipole : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemMultipole(SXF::OStream&, const char* type);

  // Destructor.
  ~UAL_SXF_ElemMultipole();

  // Open bucket: Set the deviation flag to true
  // if the bucketType is "body.dev".
  int openObject(const char* type);

  // Update bucket: Define the max order of the multipole harmonics.
  void update();

  // Zero bucket.
  void close();

  // Check the array attribute key.
  // Return SXF_TRUE or SXF_FALSE. 
  int openArray();

  // Check the array attribute key.
  // Return SXF_TRUE or SXF_FALSE. 
  int openHash();

  // Skip the LRAD attribute.
  int setScalarValue(double); 

  // Set the value at the current index in the array 
  // attribute (KL or KLS) and increment this index. 
  // Return SXF_TRUE or SXF_FALSE.
  int setArrayValue(double value);

  // Set the value at given index in the array attribute (KL or KLS).
  // Return SXF_TRUE or SXF_FALSE.
  int setHashValue(double value, int index);

  // Write data
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Get SMF element attributes
  virtual PacElemAttributes* getAttributes(const PacLattElement& element) = 0;

protected:

  // Max order of multipole harmonics.
  static int s_iMaxOrder;

  // Pointer to the Smf bucket.
  PacElemMultipole* m_pMultipole;

  // Current multipole orders.
  int m_iOrderKL, m_iOrderKTL;

  // Deviation flag.
  int m_iIsDeviation; 

protected:

  // Mlt factors.
  double* m_aMltFactor;

  // Make an array of SMF-to-SIF multipole factors. 
  void makeMltFactors();

};

#endif
