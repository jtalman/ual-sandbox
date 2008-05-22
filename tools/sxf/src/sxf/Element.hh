//# Library     : SXF
//# File        : Element.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ELEMENT_H
#define SXF_ELEMENT_H

#include "sxf/ElemBucketRegistry.hh"
#include "sxf/AcceleratorNode.hh"

namespace SXF {

/**
  The Element defines a SXF front end element reader interface. According 
  to the SXF object model, Element represents the minimal identified 
  accelerator entity. Its data are organized in the two-level structure:
  a collection of orthogonal buckets of element attributes. In the SXF front 
  end, all buckets are divided into two uneven categories: type-specific body 
  bucket and other buckets common to all element types (such as exit, entry, 
  align, and others). This design provides a very flexible approach for 
  extending element description and including new element attributes. 
  Common buckets are registered in the ElemBucketRegistry singleton that must 
  be defined in Your_Element class:
  <srcblock>
  Your_Element::Your_Element(SXF::OStream& out, const char* type, ...) 
    : SXF::Element(out, type) 
  {
    m_pCommonBuckets = Your_ElemBucketRegistry::getInstance(...);
  }
  </srcblock>
 
  In addition to the Accelerator Node header attributes, Element has the N
  complexity attribute that specifies how to split/treat the element instance 
  in diverse accelerator algorithms.  
*/
 
  class Element : public AcceleratorNode
  {
  public:

    /** Return true. */
    int isElement() const;

    /** Set the N complexity header attribute. */
    virtual void setN(double n) = 0;

    /** Select a body reader as a current bucket reader
	(Body type may be "body" or "body.dev"),
	Return true or false
    */
    virtual int openBody(const char* bodyType);

    /** Select one of common bucket readers and make it current 
	(Typical types are "exit", "exit.dev", and others),
	Return true or false
    */
    virtual int openBucket(const char* bodyType);

    /** Return a current bucket reader. */
    virtual ElemBucket* getBucket();

    /** Add a current bucket data. */
    virtual void addBucket(ElemBucket* bucket) = 0;

    /** Close a current bucket reader. */
    virtual void closeBucket();

  protected:

    /** Pointer to the element body reader. */
    ElemBucket* m_pElemBody;

    /** Pointer to the element bucket registry. */
    ElemBucketRegistry* m_pCommonBuckets;

    /** Pointer to a current body bucket reader. */
    ElemBucket* m_pElemBucket;

  protected:

    /** Constructor.*/
    Element(OStream& out, const char* elementType);

  };
}


#endif
