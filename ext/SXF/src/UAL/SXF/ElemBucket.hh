//# Library     : UAL
//# File        : UAL/SXF/ElemBucket.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_BUCKET_HH
#define UAL_SXF_ELEM_BUCKET_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /** 
   * The ElemBucket class defines a basis interface and common behaviour 
   * of SXF adaptors to the SMF collections of element attributes.
   */

  class SXFElemBucket : public SXF::ElemBucket
  {
  public:

    /** Constructor. */
    SXFElemBucket(SXF::OStream& out, const char* type, SXF::ElemBucketHash* hash);

    /** Returns true if it is deviation.
	Default: returns false.
     */
    virtual int isDeviation() const;

    /** Does nothing. Returns SXF_TRUE. */
    int openObject(const char*);

    /** Does nothing. */
    void update();

    /** Does nothing. */
    void close();

    /** Returns SXF_FALSE. */
    int openArray();

    /** Does nothing. */
    void closeArray();

    /** Returns SXF_FALSE. */
    int openHash();

    /** Does nothing. */
    void closeHash();

    /** Returns SXF_FALSE. */
    int setScalarValue(double value);

    /** Returns SXF_FALSE. */
    int setArrayValue(double value);

    /** Returns SXF_FALSE. */
    int setHashValue(double value, int index);

    /** Returns a number of SMF entry buckets.
     *  Return 0.
     */
    virtual int getEntrySize() const;

    /** Returns a number of SMF body buckets. 
     * Return 0.
     */
    virtual int getBodySize() const;

    /** Returns a number of SMF exit buckets.
     * Return 0.
     */
    virtual int getExitSize() const;

    /** Retuns a SMF entry bucket of element attributes.
     *  Returns 0.
     */
    virtual PacElemBucket* getEntryBucket(int index);

    /** Returns a SMF body bucket of element attributes.
     *  Returns 0.
     */
    virtual PacElemBucket* getBodyBucket(int index);

    /** Returns a SMF exit bucket of element attributes.
     * Returns 0.
     */
    virtual PacElemBucket* getExitBucket(int index);

    /** Writed data: Do nothing.*/
    virtual void write(ostream& out, const PacLattElement& element, const string& tab);


  };
}

#endif
