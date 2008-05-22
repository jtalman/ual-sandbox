//# Library     : SXF
//# File        : NodeRegistry.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_NODE_REGISTRY_H
#define SXF_NODE_REGISTRY_H

#include "sxf/OStream.hh"
#include "sxf/Element.hh"
#include "sxf/Sequence.hh"

namespace SXF {

  /** 
   * The NodeRegistry class implements a repository of available accelerator node 
   * readers, element and sequence readers. Selection of element readers is based
   * on the GPERF, a Perfect Hash Function Generator, written by Douglas Schmidt.
   * Registry must supply also an error element reader that is invoked when there
   * is no appropriate reader.
   */

  class NodeRegistry
  {
  public:

    /** Destructor. */
    virtual ~NodeRegistry();

    /** Return an element reader,
     * e.g. getElement("sbend")
     */
    Element* getElement(const char* elementType);

    /** Return an error element reader. */
    Element* getErrorElement();

    /** Return a sequence reader. */
    Sequence* getSequence();

  protected:

    /** Reference to an output stream. */
    OStream& m_refOStream;

    /** Number of element readers. */
    int m_iSize;
  
    /** Array of element readers. */
    Element** m_aElements;

    /** Pointer to an error element reader. */
    Element* m_pErrorElement;

    /** Pointer to a sequence reader. */
    Sequence* m_pSequence;

  protected:

    /** Constructor. */
    NodeRegistry(OStream& out);

    /** Register a particular element reader. */
    Element* bind(const char* name, Element* element);

    /** Map element types to their indecies in the collection of element readers. */
    virtual int hash(const char* type) const;

    /** Create an array of element readers. */
    void allocateRegistry();

  };

}


#endif
