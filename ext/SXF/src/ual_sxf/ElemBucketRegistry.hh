//# Library     : SXF
//# File        : ual_sxf/ElemBucketRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEM_BUCKET_REGISTRY_H
#define UAL_SXF_ELEM_BUCKET_REGISTRY_H

#include "ual_sxf/Def.hh"

//
// The ElemBucketRegistry implements a registry of SXF adaptors 
// to the SMF buckets common to all element types (entry, exit, 
// align, and others).
// 

class UAL_SXF_ElemBucketRegistry : public SXF::ElemBucketRegistry
{
public:

  // Destructor. 
  ~UAL_SXF_ElemBucketRegistry();

  // Return sigleton.
  static UAL_SXF_ElemBucketRegistry* getInstance(SXF::OStream& out); 

  // Write data.
  void write(ostream& out, const PacLattElement& element, const string& tab);

protected:

  // Singleton.
  static UAL_SXF_ElemBucketRegistry* s_pBucketRegistry;

protected:

  // Constructor.
  UAL_SXF_ElemBucketRegistry(SXF::OStream& out);

};

#endif
