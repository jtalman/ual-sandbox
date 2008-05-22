//# Library     : SXF
//# File        : echo_sxf/EchoNodeRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_NODE_REGISTRY_H
#define SXF_ECHO_NODE_REGISTRY_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /** 
   * Implements a repository of available
   * sequence and element echo writes.
   */

  class EchoNodeRegistry : public NodeRegistry
  {
  public: 

    /** Destructor. */
    ~EchoNodeRegistry();

    /** Return a singleton. */
    static EchoNodeRegistry* getInstance(OStream& out);   

  protected:

    /** Singleton. */
    static EchoNodeRegistry* s_pNodeRegistry;

  protected:

    /** Constructor. */
    EchoNodeRegistry(OStream& out);  

  };
}

#endif
