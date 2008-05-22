//# Library     : UAL
//# File        : ElemMltBody.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_BODY_HH
#define UAL_SXF_ELEM_MLT_BODY_HH

#include "SMF/PacElemBend.h"
#include "UAL/SXF/buckets/ElemMultipole.hh"

namespace UAL {

  /**
   * The ElemMltBody is the SXF adaptor to the SMF collection of
   * body multipole attributes: Multipole bucket (kl, ktl). 
   */

  class SXFElemMltBody : public SXFElemMultipole
  {
  public:

    /** Constructor. */
    SXFElemMltBody(SXF::OStream& out);

    /** Destructor. */
    ~SXFElemMltBody();

    /** Zero bucket. */
    void close();  

    /** Returns 2 (Bend and Multipole). */
    int getBodySize() const;

    /** Returns the SMF body bucket selected by the given index (0 or 1).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);
  
    /** Sets the value at the current index in the array 
     * attribute (KL or KLS) and increment this index. 
     * Skips the attribute if it is KL[0] and is not a deviation.
     */
    int setArrayValue(double value);

    /** Sets the value at given index in the array attribute (KL or KLS).
     * Skips the attribute if it is KL[0] and is not a deviation.
     */  
    int setHashValue(double value, int index);

    /** Writes data */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Returns body attributes. 
    PacElemAttributes* getAttributes(const PacLattElement& element);
    */

    /** Returns design multipole attributes */
    virtual PacElemMultipole getDesignMultipole(const PacLattElement& element);

    /** Returns total multipole attributes  */
    virtual PacElemMultipole getTotalMultipole(const PacLattElement& element);

  protected:

    //  Thin Dipole 

    /** Pointer to the SMF bucket. */
    PacElemBend* m_pBend;

    /**  Bend status. */
    int m_iBendStatus;

  };
}

#endif
