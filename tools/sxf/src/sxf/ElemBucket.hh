//# Library     : SXF
//# File        : ElemBucket.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_BUCKET_H
#define SXF_ELEM_BUCKET_H

#include "sxf/Def.hh"
#include "sxf/OStream.hh"
#include "sxf/hashes/ElemBucketHash.hh"

namespace SXF {

  /** 
   * The ElemBucket class defines an interface to element buckets, orthogonal 
   * collections of element attributes. Physically, these collections are 
   * implemented by the ElemBucketHash instances, hashes of attributes key and 
   * values. The SXF front end provides hashes for most SIF/MAD elements. However,
   * this list can be extended for new projects.
   * 
   * The element attributes values may be one of two types: numeric literal and
   * numeric literal array.
   */ 
 
  class ElemBucket
  {
  public:  

    /** Destructor. */
    virtual ~ElemBucket();

    /** Prepare a bucket reader for operations,
     * Return true or false.
     */
    virtual int openObject(const char* bucketType) = 0;

    /** Complete all operations. */
    virtual void update() = 0;

    /** Return a bucket reader to its initial conditions. */
    virtual void close() = 0;

    /** Select a bucket attribute and make it current. */
    virtual int openAttribute(const char* name);

    /** Close a current attribute. */
    virtual void closeAttribute();

    /** Open an array of attribute values. */
    virtual int openArray() = 0;  

    /** Close an array of attribute values. */
    virtual void closeArray() = 0;  

    /** Open a hash of attribute values. */
    virtual int openHash() = 0;  

    /** Close a hash of attribute values. */
    virtual void closeHash() = 0;  

    /** Define a scalar value. */
    virtual int setScalarValue(double value) = 0;

    /** Define an array element value. */
    virtual int setArrayValue(double value) = 0;

    /** Define a hash element value. */
    virtual int setHashValue(double value, int index) = 0;

  protected:

    /** Reference to the error stream. */
    OStream& m_refOStream;

    /** Bucket type. */
    char* m_sType;

    /** Index of a current attribute. */
    int m_iAttribID;

    /** Hash of bucket attributes. */
    ElemBucketHash* m_pHash;

  protected:

    /** Constructor. */
    ElemBucket(OStream& out, const char* bucketType, ElemBucketHash* hash);

    /** Set type. */
    void setType(const char* bucketType);

  };
}

#endif

