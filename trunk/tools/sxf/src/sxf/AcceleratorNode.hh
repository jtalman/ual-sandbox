//# Library     : SXF
//# File        : AcceleratorNode.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ACCELERATOR_NODE_H
#define SXF_ACCELERATOR_NODE_H

#include "sxf/Def.hh"
#include "sxf/OStream.hh"

namespace SXF {

  /** 
   * The AcceleratorNode class defines the common basis interface of SXF 
   * sequence and element readers. According to the SXF object model, 
   * Accelerator is represented by a hierarchical tree of sequences and elements.
   * Each accelerator node has a unique identifier with the particular 
   * accelerator namespace, but it may be shared by the different accelerators
   * (e.g. in iteraction regions, injection and extraction systems, and others).
   * All its attributes are optional and have default value or behaviour.
   */

  class AcceleratorNode
  {
  public:

    /** Destructor. */
    virtual ~AcceleratorNode();

    /** Prepare a node reader for operations. */
    virtual int openObject(const char* nodeName, const char* nodeType) = 0; 

    /** Complete all operations 
     * (Default: do nothing)
     */ 
    virtual void update();

    /** Return this node reader to its initial conditions (Default: do nothing)
     */
    virtual void close(); 

    /** Set a reference to the design entity. Because the SXF accelerator
     * description is complete and independent from the external sources,
     * this link does not effect on the sequence parameters and provides 
     * only the relationship with the site-specific accelerator design
     * model.
     */
    virtual void setDesign(const char* name) = 0;

    /** Set an node length along the design orbit (units: m). */
    virtual void setLength(double l) = 0;

    /** Set a longitudinal position of the node with respect to the beginning 
     * of the parent sequence (units: m).
     */
    virtual void setAt(double l) = 0;

    /** Set a horizontal angle (units: rad). */
    virtual void setHAngle(double h) = 0;

    /** Check node if it is a leaf or a composite. */
    virtual int isElement() const = 0;

  protected:

    /** Reference to the output stream */
    OStream& m_refOStream;

    /** Node type */
    char* m_sType;

  protected:

    /** Constructor. */
    AcceleratorNode(OStream& out, const char* nodeType);

    /** Set node type */
    void setType(const char* type);  

  };
}

#endif
