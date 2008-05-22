//# Library     : UAL
//# File        : ElemBend.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_BEND_HH
#define UAL_SXF_ELEM_BEND_HH

#include "SMF/PacElemBend.h"
#include "SMF/PacElemMultipole.h"
#include "UAL/SXF/ElemBucket.hh"

namespace UAL {

  /** 
   * The ElemBend class is the SXF adaptor to the SMF collections
   * of bend attributes: Bend (andle, fint) and Multipole (kl, ktl)
   * buckets. The SXF bend attributes are represented by scalar 
   * attributes (FINT, HGAP, E1, E2) and array attributes (KL, KLS).
   */


  class SXFElemBend : public SXFElemBucket
  {
  public:

    /**  Constructor. */
    SXFElemBend(SXF::OStream& out);

    /** Destructor. */
    ~SXFElemBend();

    /** Retuns if it is deviation */
    int isDeviation() const;

    /** Opens bucket: Set the deviation flag to true 
     * if the bucketType is "body.dev".
     */
    int openObject(const char* bucketType);

    /** Updates bucket: Define the max order of the multipole harmonics. */
    void update();

    /** Zero bucket. */
    void close();

    /** Checks the array attribute key (KL or KLS). 
     * Returns SXF_TRUE or SXF_FALSE.
     */
    int openArray();

    /** Does nothing.*/
    void closeArray();

    /** Checks the array attribute key (KL or KLS). 
     * Returns SXF_TRUE or SXF_FALSE. 
     */
    int openHash();

    /** Does nothing.*/
    void closeHash();  

    /** Checks the scalar attribute key (FINT, E1, E2) and set its value. 
     * Skip the HGAP attribute.
     */
    int setScalarValue(double value);

    /** Set the value at the current index in the array 
     * attribute (KL or KLS) and increment this index. 
     * Define the SMF bend angle if it is KL[0] and is
     * not a deviation.
     */
    int setArrayValue(double value);

    /** Sets the value at given index in the array attribute (KL or KLS).
     * Define the SMF bend angle if it is KL[0] and is not a deviation.
     */
    int setHashValue(double value, int index);

    /** Returns 1 because the SXF E1 attribute is represented
     * by the SMF entry bucket attribute ANGLE.
     */
    int getEntrySize() const;

    /** Returns 2 because the SXF bend body attributes are represented 
     * by two SMF body buckets: Bend and Multipole.
     */
    int getBodySize() const;

    /** Returns 1 because the SXF E2 attribute is represented
     * by the SMF exit bucket attribute ANGLE.
     */
    int getExitSize() const;

    /** Returns the SMF entry bucket selected by the given index (0).
     * Return 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getEntryBucket(int index);

    /** Returns the SMF body bucket selected by the given index (0 or 1).
     * Returns 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getBodyBucket(int index);

    /** Returns the SMF exit bucket selected by the given index (0).
     * Return 0, if the bucket is not defined or empty.
     */
    PacElemBucket* getExitBucket(int index);
  
    /** Writes data.*/
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    // Bend data 

    /** Deviation flag */
    int m_iIsDeviation; 
  
    /** Pointer to the SMF buckets (entry, body, and exit) */
    PacElemBend* m_pBend[3]; 

    /** Bend status (defined or empty) */
    int m_iBendStatus[3];

    // Multipole data

    /** Pointer to the SMF multipole bucket  */       
    PacElemMultipole* m_pMultipole;   

    /** Max order of multipole harmonics */
    static int s_iMaxOrder;

    /** Current multipole orders */
    int m_iOrderKL, m_iOrderKTL;

  protected:

    /** Mlt factors */
    double* m_aMltFactor;

    /** Make an array of SMF-to-SIF multipole factors. */
    void makeMltFactors();

  };

}

#endif
