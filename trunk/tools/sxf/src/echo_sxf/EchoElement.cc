#include "echo_sxf/EchoElement.hh"
#include "echo_sxf/EchoElemBucketRegistry.hh"

// Constructor.
SXF::EchoElement::EchoElement(SXF::OStream& out, const char* type, SXF::ElemBucket* body)
  : SXF::Element(out, type)
{
  m_pElemBody = body;
  m_pCommonBuckets = SXF::EchoElemBucketRegistry::getInstance(out);
}

// Destructor.
SXF::EchoElement::~EchoElement()
{  
  if(m_pElemBody) { delete m_pElemBody; }
}

// Do nothing. Return SXF_TRUE.
int SXF::EchoElement::openObject(const char*, const char*)
{
  return SXF_TRUE;
}

// Do nothing.
void SXF::EchoElement::update()
{
}

// Do nothing.
void SXF::EchoElement::close()
{
}

// Do nothing. 
void SXF::EchoElement::setDesign(const char*)
{
}

// Do nothing.
void SXF::EchoElement::setLength(double )
{
}

// Do nothing.
void SXF::EchoElement::setAt(double)
{
}

// Do nothing.
void SXF::EchoElement::setHAngle(double)
{
}

// Do nothing.
void SXF::EchoElement::setN(double)
{
}

// Do nothing
void SXF::EchoElement::addBucket(SXF::ElemBucket*) 
{
}

