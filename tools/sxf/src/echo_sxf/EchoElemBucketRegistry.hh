//# Library     : SXF
//# File        : echo_sxf/EchoElemBucketRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ELEM_BUCKET_REGISTRY_H
#define SXF_ECHO_ELEM_BUCKET_REGISTRY_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /** 
   * Implements a registry of element buckets
   * common to all element types (entry, exit, align, and others).
   */

  class EchoElemBucketRegistry : public ElemBucketRegistry
  {
  public:

    /** Destructor. */
    ~EchoElemBucketRegistry();

    /** Return a sigleton. */
    static EchoElemBucketRegistry* getInstance(OStream& out); 

  protected:

    /** Singleton. */
    static EchoElemBucketRegistry* s_pBucketRegistry;

  protected:

    /** Constructor. */
    EchoElemBucketRegistry(OStream& out);

  };
}

#endif
