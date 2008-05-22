//# Library     : UAL
//# File        : ElemMltEntry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_ENTRY_HH
#define UAL_SXF_ELEM_MLT_ENTRY_HH

#include "UAL/SXF/buckets/ElemMultipole.hh"

namespace UAL {

  /**
   * The ElemMltEntry is the SXF adaptor to the SMF collection of
   * entry multipole attributes: Multipole bucket (kl, ktl). 
   */

  class SXFElemMltEntry : public SXFElemMultipole
  {
  public:

    /** Constructor. */
    SXFElemMltEntry(SXF::OStream& out);

    /** Opens bucket: Set the deviation flag to true
     * if the bucketType is "entry.dev".
     */
    int openObject(const char* type);

    /** Returns 1. */
    int getEntrySize() const;

    /** Returns the SMF entry bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getEntryBucket(int index);

  protected:
  
    /* Returns entry attributes 
    PacElemAttributes* getAttributes(const PacLattElement& element);
    */

    /** Returns design multipole attributes */
    virtual PacElemMultipole getDesignMultipole(const PacLattElement& element);

    /** Returns total multipole attributes  */
    virtual PacElemMultipole getTotalMultipole(const PacLattElement& element);

  };

}

#endif
