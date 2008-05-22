//# Library     : UAL
//# File        : UAL/SXF/NodeRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky


#ifndef UAL_SXF_NODE_REGISTRY_HH
#define UAL_SXF_NODE_REGISTRY_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /** 
   * The NodeRegistry class is a repository of the SXF
   * element and sequence adaptors.   
   */

  class SXFNodeRegistry : public SXF::NodeRegistry
  {
  public: 

    /** Destructor. */
    ~SXFNodeRegistry();

    /** Return a singleton. */
    static SXFNodeRegistry* getInstance(SXF::OStream& out,  PacSmf& smf);  

  protected:

    /** Singleton. */
    static SXFNodeRegistry* s_pNodeRegistry;

  protected:

    /** Constructor. */
    SXFNodeRegistry(SXF::OStream& out, PacSmf& smf);

  };

}

#endif
