//# Library     : UAL
//# File        : ElemMultipole.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_MULTIPOLE_HH
#define UAL_SXF_ELEM_MULTIPOLE_HH

#include "SMF/PacElemMultipole.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemMultipole class defines a basis interface and
   * a common behavior of the SMF adaptors to the SMF 
   * collections of the multipole attributes.
   */

  class SXFElemMultipole : public SXFElemBucket
  {
  public:

    /** Constructor.*/
    SXFElemMultipole(SXF::OStream&, const char* type);

    /** Destructor. */
    ~SXFElemMultipole();

    /** Retuns if it is deviation */
    int isDeviation() const;

    /** Opens bucket: Set the deviation flag to true
     * if the bucketType is "body.dev".
     */
    int openObject(const char* type);

    /** Updates bucket: Define the max order of the multipole harmonics.*/
  void update();

    /** Zero bucket.*/
    void close();

    /** Checks the array attribute key.
     * Returns SXF_TRUE or SXF_FALSE. 
     */
    int openArray();

    /** Checks the array attribute key.
     * Returns SXF_TRUE or SXF_FALSE. 
     */
    int openHash();

    /** Skips the LRAD attribute.*/
    int setScalarValue(double); 

    /** Sets the value at the current index in the array 
     * attribute (KL or KLS) and increment this index. 
     * Returns SXF_TRUE or SXF_FALSE.
     */
    int setArrayValue(double value);

    /** Sets the value at given index in the array attribute (KL or KLS).
     * Returns SXF_TRUE or SXF_FALSE.
     */
    int setHashValue(double value, int index);

    /** Writes data */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /* Returns SMF element attributes 
    virtual PacElemAttributes* getAttributes(const PacLattElement& element) = 0;
    */

    /** Returns design multipole attributes */
    virtual PacElemMultipole getDesignMultipole(const PacLattElement& element) = 0;

    /** Returns total multipole attributes  */
    virtual PacElemMultipole getTotalMultipole(const PacLattElement& element) = 0;

    /** Writes attributes */
    void writeMultipole(ostream& out, PacElemMultipole& multipole, int isDev, const string& tab);

  protected:

    /** Max order of multipole harmonics. */
    static int s_iMaxOrder;

    /** Pointer to the Smf bucket. */
    PacElemMultipole* m_pMultipole;

    /** Current multipole orders. */
    int m_iOrderKL, m_iOrderKTL;

    /** Deviation flag. */
    int m_iIsDeviation; 

  protected:

    /** Mlt factors. */
    double* m_aMltFactor;

    /** Make an array of SMF-to-SIF multipole factors. */
    void makeMltFactors();

  };
}

#endif
