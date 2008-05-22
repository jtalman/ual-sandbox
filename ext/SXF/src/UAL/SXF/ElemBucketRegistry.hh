//# Library     : UAL
//# File        : UAL/SXF/ElemBucketRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_BUCKET_REGISTRY_HH
#define UAL_SXF_ELEM_BUCKET_REGISTRY_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /**
   * The ElemBucketRegistry implements a registry of SXF adaptors 
   * to the SMF buckets common to all element types (entry, exit, 
   * align, and others).
   */

  class SXFElemBucketRegistry : public SXF::ElemBucketRegistry
  {
  public:

    /** Destructor. */
    ~SXFElemBucketRegistry();

    /** Returns sigleton. */
    static SXFElemBucketRegistry* getInstance(SXF::OStream& out); 

    /** Writes data. */
    void write(ostream& out, const PacLattElement& element, const string& tab);

  protected:

    /** Singleton.*/
    static SXFElemBucketRegistry* s_pBucketRegistry;

  protected:

    /** Constructor.*/
    SXFElemBucketRegistry(SXF::OStream& out);

  };

}

#endif
