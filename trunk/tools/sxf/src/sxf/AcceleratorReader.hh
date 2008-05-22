//# Library     : SXF
//# File        : AcceleratorReader.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ACCELERATOR_READER_H
#define SXF_ACCELERATOR_READER_H

#include "sxf/OStream.hh"
#include "sxf/NodeRegistry.hh"
#include "sxf/SequenceStack.hh"

namespace SXF {

  /** 
   * The AcceleratorReader class defines the SXF front end accelerator reader 
   * interface. According to the SXF object model, Accelerator is a root 
   * node in the hierarchical tree of sequences and elements. The SXF 
   * supports all SIF/MAD element types. However, the SXF open model 
   * allows description (and inclusion) of arbitrary element types (e.g.
   * Muon Collider Ionization Cooling, CESR superimposed solenoid and 
   * quadrupole elements, LHC and CESR parasitic beam-beam effects, and
   * others). To support this feature, the front end framework provides 
   * the node registry that can be adapted to particular project, site, 
   * or accelerator program. This registry must be defined in the 
   * Your_Accelerator class:
   * <srcblock>
   * Your_AcceleratorReader::Your_AcceleratorReader(SXF::OStream& out, ...) 
   * : SXF::AcceleratorReader(out) 
   * {
   *  m_pNodeRegistry = Your_NodeRegistry::getInstance(...);
   * }
   * </srcblock>
   */

  class AcceleratorReader
  {
  public:
 
    /** Destructor. */
    virtual ~AcceleratorReader();

    /** Prepare an accelerator reader for operations, 
     * Push a root sequence reader to the stack and make it current,
     * Return true or false.
     */
    virtual int open(const char* acceleratorName);

    /** Return an accelerator reader to its initial conditions. */
    virtual void close();

    /** Push a sequence reader to the stack and make it current,
     * Return true  or false.
     */ 
    virtual int openSequence(const char* sequenceName);

    /** Return a current sequence reader. */
    virtual Sequence* getSequence();

    /** Close a current sequence reader and pop it from the stack. */
    virtual void closeSequence();  

    /** Select an element reader and make it current,
     * Return true or false.
     */
    virtual int openElement(const char* elementType, const char* elementName);

    /** Return a current element reader. */
    virtual Element* getElement();

    /** Close a current element reader. */
    virtual void closeElement();

    /** Return OStream. */
    OStream& echo();

  public:

    /** Global reader shared by different objects */
    static AcceleratorReader* s_reader;

  protected:

    /** Reference to OStream. */
    OStream& m_refOStream;

    /** Pointer to the accelerator node registry. */
    NodeRegistry* m_pNodeRegistry;

    /** Pointer to a current sequence reader. */
    Sequence* m_pSequence;

    /** Stack of sequences. */
    SequenceStack m_SequenceStack;

    /** Pointer to a current element reader. */
    Element* m_pElement;

  protected:

    /** Constructor. */
    AcceleratorReader(OStream& out, int stackSize = 20);

  };

}

#endif
