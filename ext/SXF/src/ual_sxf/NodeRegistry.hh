//# Library     : SXF
//# File        : ual_sxf/NodeRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky


#ifndef UAL_SXF_NODE_REGISTRY_H
#define UAL_SXF_NODE_REGISTRY_H

#include "ual_sxf/Def.hh"

//
// The NodeRegistry class is a repository of the SXF
// element and sequence adaptors.   
// 

class UAL_SXF_NodeRegistry : public SXF::NodeRegistry
{
public: 

 // Destructor.
  ~UAL_SXF_NodeRegistry();

  // Return a singleton.
  static UAL_SXF_NodeRegistry* getInstance(SXF::OStream& out,  PacSmf& smf);  

protected:

  // Singleton.
  static UAL_SXF_NodeRegistry* s_pNodeRegistry;

protected:

  // Constructor.
  UAL_SXF_NodeRegistry(SXF::OStream& out, PacSmf& smf);


};

#endif
