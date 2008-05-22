#include "sxf/Element.hh"

// Constructor.
SXF::Element::Element(SXF::OStream& out, const char* type) 
  : 
  SXF::AcceleratorNode(out, type),
  m_pElemBody(0),
  m_pCommonBuckets(0),  
  m_pElemBucket(0)
{
}

// Return SXF_TRUE.
int SXF::Element::isElement() const
{
  return SXF_TRUE;
}

// Select a body reader as a current bucket reader.
int SXF::Element::openBody(const char* type)
{
  m_pElemBucket = m_pElemBody;
  return m_pElemBucket->openObject(type);
}

// Select one of common bucket readers and make it current.
int SXF::Element::openBucket(const char* type)
{
  m_pElemBucket = m_pCommonBuckets->getBucket(type);
  if(!m_pElemBucket) {  
    m_pElemBucket = m_pCommonBuckets->getErrorBucket(); 
    m_pElemBucket->openObject(type);
    return SXF_FALSE;
  }
  return m_pElemBucket->openObject(type);
}

// Return a current bucket reader.
SXF::ElemBucket* SXF::Element::getBucket()
{
  return m_pElemBucket;
}

// Close a current bucket reader.
void SXF::Element::closeBucket() 
{
  m_pElemBucket->update();
  addBucket(m_pElemBucket);
  m_pElemBucket->close();
  m_pElemBucket = 0;
}
