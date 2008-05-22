//# Library     : UalSXF
//# File        : ElemAperture.hh
//# Copyright   : see Copyrigh file
//# Author      : Raymond Fliller 

#ifndef UAL_SXF_ELEM_APERTURE_H
#define UAL_SXF_ELEM_APERTURE_H

#include "SMF/PacElemAperture.h"
#include "ual_sxf/ElemBucket.hh"

//
// The ElemAperture class is the SXF adaptor to the SMF 
// Aperture bucket (shape, xsize, ysize). The SXF collimator 
// attributes are xsize and ysize. 
//

class UAL_SXF_ElemAperture : public UAL_SXF_ElemBucket
{
public:

  // Constructor.
  UAL_SXF_ElemAperture(SXF::OStream& out);

  // Destructor.
  ~UAL_SXF_ElemAperture();

  // Zero bucket.
  void close();  

  // Check the scalar attribute key (XSIZE, YSIZE) and set its value. 
  // Return SXF_TRUE or SXF_FALSE.
  int setScalarValue(double value);

  // Return 1 because the SXF collimator body attributes are 
  // represented by the one SMF body bucket: Aperture.
  int getBodySize() const;

  // Get the SMF body bucket selected by the given index (0).
  // Return 0, if the bucket is not defined.
  PacElemBucket* getBodyBucket(int index);

  // Write data
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Pointer to the Smf bucket.
  PacElemAperture* m_pAperture;

};

#endif
