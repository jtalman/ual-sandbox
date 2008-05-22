//# Library     : SXF
//# File        : ElementBucketRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEM_BUCKET_REGISTRY_H
#define SXF_ELEM_BUCKET_REGISTRY_H

#include "sxf/OStream.hh"
#include "sxf/ElemBucket.hh"

namespace SXF {

  /** 
   * The ElemBucketRegistry class represents a collection of element buckets 
   * common to all element types (such as exit, entry, align, and others).
   * Registry must supply also an error element bucket reader that is invoked 
   * when there is no appropriate reader.
   */

  class ElemBucketRegistry
  {
  public:

    /** Destructor. */
    virtual ~ElemBucketRegistry();

    /** Return a bucket reader, 
     * e.g. getBucket("aperture").
     */
    ElemBucket* getBucket(const char* type);

    /** Return an error element bucket reader. */
    ElemBucket* getErrorBucket();

  protected:

    /** Reference to the output stream. */
    OStream& m_refOStream;

    /** Number of bucket readers. */
    int m_iSize;

    /** Array of bucket readers. */
    ElemBucket** m_aBuckets;

    /** Pointer to a error element bucket reader. */
    ElemBucket* m_pErrorBucket;

  protected:

    /** Constructor(s) */
    ElemBucketRegistry(OStream& out);

    /** Register a particular bucket reader,
     * e.g. bindBucket("aperture", new <prefix>_SXF_ElemAperture(<arguments>))
     */
    ElemBucket* bind(const char* name, ElemBucket* bucket);

    /** Map bucket types to their indecies in the array of bucket readers. */
    virtual int hash(const char* type) const;

    /** Allocate an array of element bucket readers. */
    void allocateRegistry();

  };

}

#endif
