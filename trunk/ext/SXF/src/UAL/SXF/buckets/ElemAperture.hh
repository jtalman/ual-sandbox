//# Library     : UAL
//# File        : ElemAperture.hh
//# Copyright   : see Copyrigh file
//# Author      : Raymond Fliller 

#ifndef UAL_SXF_ELEM_APERTURE_HH
#define UAL_SXF_ELEM_APERTURE_HH

#include "SMF/PacElemAperture.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemAperture class is the SXF adaptor to the SMF 
   * Aperture bucket (shape, xsize, ysize). The SXF collimator 
   * attributes are xsize and ysize. 
   */

  class SXFElemAperture : public SXFElemBucket
  {
  public:

    /** Constructor. */
    SXFElemAperture(SXF::OStream& out);

    /** Destructor. */
    ~SXFElemAperture();

    /** Zero bucket. */
    void close();  

    /** Check the scalar attribute key (XSIZE, YSIZE) and set its value. 
     * Return SXF_TRUE or SXF_FALSE.
     */
    int setScalarValue(double value);

    /** Returns 1 because the SXF collimator body attributes are 
     * represented by the one SMF body bucket: Aperture.
     */
    int getBodySize() const;

    /** Returns the SMF body bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined.
     */
    PacElemBucket* getBodyBucket(int index);

    /** Write data */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Pointer to the Smf bucket. */
    PacElemAperture* m_pAperture;

  };

}

#endif
