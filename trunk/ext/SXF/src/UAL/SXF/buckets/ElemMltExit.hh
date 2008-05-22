//# Library     : UAL
//# File        : ElemMltExit.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MLT_EXIT_HH
#define UAL_SXF_ELEM_MLT_EXIT_HH

#include "UAL/SXF/buckets/ElemMultipole.hh"

namespace UAL {

  /** 
   * The ElemMltExit is the SXF adaptor to the SMF collection of
   * exit multipole attributes: Multipole bucket (kl, ktl). 
   */

  class SXFElemMltExit : public SXFElemMultipole
  {
  public:

    /** Constructor. */
    SXFElemMltExit(SXF::OStream& out);

    /** Opens bucket: Set the deviation flag to true
     * if the bucketType is "exit.dev".
     */
    int openObject(const char* type);

    /** Returns 1. */
    int getExitSize() const;

    /** Returns the SMF exit bucket selected by the given index (0).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getExitBucket(int index);

  protected:
  
    /* Returns exit attributes 
    PacElemAttributes* getAttributes(const PacLattElement& element);
    */

    /** Returns design multipole attributes */
    virtual PacElemMultipole getDesignMultipole(const PacLattElement& element);

    /** Returns total multipole attributes  */
    virtual PacElemMultipole getTotalMultipole(const PacLattElement& element);

  };
}

#endif
