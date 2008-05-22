//# Library     : UAL
//# File        : ElemAlign.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_ALIGN_HH
#define UAL_SXF_ELEM_ALIGN_HH

#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemAlign class is the SXF adaptor to the SMF collections
   * of align attributes: Offset (dx, dy, ds) and Rotation (dphi, 
   * dtheta, tilt) buckets. The SXF align attributes are represented
   * by the AL array, [dx, dy, ds,  dphi, dtheta, tilt].
   */

  class SXFElemAlign : public SXFElemBucket
  {
  public:

    /** Constructor. */
    SXFElemAlign(SXF::OStream&);

    /** Destructor. */
    ~SXFElemAlign();

    /** Retuns if it is deviation */
    int isDeviation() const;

    /** Opens bucket: Set the deviation flag to true
     * if the bucketType is "align.dev".
     */
    int openObject(const char* type);

    /** Zero bucket.*/
    void close();

    /** Checks the attribute key.
     * Return SXF_TRUE or SXF_FALSE.
     */
    int openArray();

    /** Does nothing. */
    void closeArray();

    /** Does nothing. Return SXF_TRUE. */
    int openHash();

    /**  Does nothing. */
    void closeHash();  

    /** Sets the value at the current index in the array attribute (AL)
     * and increment this index. Return SXF_TRUE or SXF_FALSE.
     */
    int setArrayValue(double value);

    /** Sets the value at given index in the array attribute (AL).
     * Returns SXF_TRUE or SXF_FALSE.
     */
     int setHashValue(double value, int index);

    /** Returns 2 because the align attributes are represented by two 
     * SMF buckets: Offset and Rotation.
     */
    int getBodySize() const;

    /** Returns the SMF body bucket selected by the given index (0 or 1).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);

    /** Writes data. */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Current index in the multi-valued indexed attribute.*/
    int m_iIndex;

    /** Bucket status (defined or empty).*/
    int m_iOffsetStatus;
    int m_iRotationStatus;

    /** Pointer to the SMF buckets.*/
    PacElemOffset*   m_pOffset;
    PacElemRotation* m_pRotation;

    /** Deviation flag. */
    int m_iIsDeviation; 


  };

}

#endif
