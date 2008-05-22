//# Library     : UAL
//# File        : ElemKicker.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_KICKER_HH
#define UAL_SXF_ELEM_KICKER_HH

#include "SMF/PacElemMultipole.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /**
   * The ElemKicker is the SXF adaptor to the SMF collection of
   * kicker attributes: Multipole (kl, ktl) bucket. The SXF kicker
   * attributes are represented by two scalars, kl and kls.
   */

  class SXFElemKicker : public SXFElemBucket
  {
  public:

    /** Constructor. */
    SXFElemKicker(SXF::OStream&);

    /** Destructor. */
    ~SXFElemKicker();

    /** Zero bucket. */
    void close();

    /** Checks the scalar attribute key (KL , KLS) and set its value. 
     * Returns SXF_TRUE or SXF_FALSE.
     */
    int setScalarValue(double); 

    /** Writes data. */
    void write(ostream& out, const PacLattElement& element, const string& tab);

    /** Returns 1 because the SXF kicker attributes are represented
     * by one SMF Multipole bucket.
     */
    int getBodySize() const;

    /** Returns the SMF body bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);


  protected:

    /** Bucket status (defined or empty) */
    int m_iMltStatus;

    /** Pointer to the Smf bucket */
    PacElemMultipole* m_pMultipole;

  };
}

#endif
