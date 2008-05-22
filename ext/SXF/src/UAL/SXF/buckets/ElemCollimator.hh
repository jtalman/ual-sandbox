//# Library     : UAL
//# File        : ElemCollimator.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_COLLIMATOR_HH
#define UAL_SXF_ELEM_COLLIMATOR_HH

#include "SMF/PacElemAperture.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemCollimator class is the SXF adaptor to the SMF 
   * Aperture bucket (shape, xsize, ysize). The SXF collimator 
   * attributes are xsize and ysize. 
   */

  class SXFElemCollimator : public SXFElemBucket
  {
  public:

    /** Constructor.*/
    SXFElemCollimator(SXF::OStream& out, const char* type, const char shape);

    /** Destructor. */
    ~SXFElemCollimator();

    /** Updates bucket: Defines the aperture shape.*/
    void update();

    /** Zero bucket.*/
    void close();  

    /** Checks the scalar attribute key (XSIZE, YSIZE) and set its value. 
     * Returns SXF_TRUE or SXF_FALSE.
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

    /**  Writes data */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Aperture shape */
    char m_cShape;

    /** Pointer to the Smf bucket. */
    PacElemAperture* m_pAperture;

  };

}

#endif
