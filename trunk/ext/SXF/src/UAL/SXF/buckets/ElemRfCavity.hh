//# Library     : UAL
//# File        : ElemRfCavity.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_RFCAVITY_HH
#define UAL_SXF_ELEM_RFCAVITY_HH

#include "SMF/PacElemRfCavity.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemRfCavity is the SXF adaptor to the SMF collection of
   * rf. cavity attributes: RfCavity bucket (volt, lag, harmon). 
   * The SXF rf cavity scalar attributes are volt, lag, harmon, shunt,
   * and tfill.
   */

  class SXFElemRfCavity : public SXFElemBucket
  {
  public:

    /** Constructor. */
    SXFElemRfCavity(SXF::OStream& out);

    /** Destructor. */
    ~SXFElemRfCavity();

    /** Zero bucket. */
    void close();

    /** Checks the scalar attribute key (VOLT, LAG, HARMON) and set its value. 
     * Skips the SHUNT and TFILL attribute.
     */
    int setScalarValue(double value);

    /** Returns 1 because the SXF rf attributes are represented by
     * one SMF bucket.
     */
    int getBodySize() const;

    /** Returns the SMF rf cavity bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);
  
    /** Writes data. */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Bucket status (defined or empty) */
    int m_iRfStatus;

    /** Pointer to the Smf bucket. */
    PacElemRfCavity* m_pRfCavity;

  };
}

#endif
