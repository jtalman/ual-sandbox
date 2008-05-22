//# Library     : UalSXF
//# File        : ual_sxf/ElemBucket.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_BUCKET_H
#define UAL_SXF_ELEM_BUCKET_H

#include "ual_sxf/Def.hh"

//
// The ElemBucket class defines a basis interface and common behaviour 
// of SXF adaptors to the SMF collections of element attributes.
// 

class UAL_SXF_ElemBucket : public SXF::ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemBucket(SXF::OStream& out, const char* type, SXF::ElemBucketHash* hash);

  // Do nothing. Return SXF_TRUE.
  int openObject(const char*);

  // Do nothing. 
  void update();

  // Do nothing.
  void close();

  // Return SXF_FALSE.
  int openArray();

  // Do nothing. 
  void closeArray();

  // Return SXF_FALSE.
  int openHash();

  // Do nothing.
  void closeHash();

  // Return SXF_FALSE.
  int setScalarValue(double value);

  // Return SXF_FALSE.
  int setArrayValue(double value);

  // Return SXF_FALSE.
  int setHashValue(double value, int index);

  // Get a number of SMF entry buckets. 
  // Return 0.
  virtual int getEntrySize() const;

  // Get a number of SMF body buckets. 
  // Return 0.
  virtual int getBodySize() const;

  // Get a number of SMF exit buckets.
  // Return 0.
  virtual int getExitSize() const;

  // Get a SMF entry bucket of element attributes.
  // Return 0.
  virtual PacElemBucket* getEntryBucket(int index);

  // Get a SMF body bucket of element attributes.
  // Return 0.
  virtual PacElemBucket* getBodyBucket(int index);

  // Get a SMF exit bucket of element attributes.
  // Return 0.
  virtual PacElemBucket* getExitBucket(int index);

  // Write data: Do nothing.
  virtual void write(ostream& out, const PacLattElement& element, const string& tab);


};

#endif
