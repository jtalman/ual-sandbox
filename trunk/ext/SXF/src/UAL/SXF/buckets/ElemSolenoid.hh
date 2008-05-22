//# Library     : UAL
//# File        : ElemSolenoid.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_SOLENOID_HH
#define UAL_SXF_ELEM_SOLENOID_HH

#include "SMF/PacElemSolenoid.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemSolenoid is the SXF adaptor to the SMF collection of
   * solenoid attributes: Solenoid bucket (ks). The SXF solenoid 
   * scalar attributes are ks.
   */

  class SXFElemSolenoid : public SXFElemBucket
  {
  public:

    /** Constructor. */
    SXFElemSolenoid(SXF::OStream& out);

    /** Destructor. */
    ~SXFElemSolenoid();

    /** Zero bucket. */
    void close(); 
 
    /** Checks the scalar attribute key (KS) and set its value. */
    int setScalarValue(double value);

    /** Returns 1  because the SXF  solenoid attributes are represented by
     * one SMF bucket.
     */
    int getBodySize() const;

    /** Returns the SMF solenoid bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);

    /** Writes data. */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Bucket status (defined or empty) */
    int m_iSolenoidStatus;

    /** Pointer to the Smf bucket */
    PacElemSolenoid* m_pSolenoid;

  };
}

#endif
