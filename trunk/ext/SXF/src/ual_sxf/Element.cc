#include "SMF/PacElemLength.h"
#include "SMF/PacElemComplexity.h"

#include "ual_sxf/ElemBucketRegistry.hh"
#include "ual_sxf/ElemBucket.hh"
#include "ual_sxf/Element.hh"

#include "ual_sxf/Perfect_Hash.hh"

// Constructor.
UAL_SXF_Element::UAL_SXF_Element(SXF::OStream& out, const char* type, 
				 SXF::ElemBucket* bodyBucket, PacSmf& smf)
  : SXF::Element(out, type),
    m_dL(0), m_dAt(0), m_dHAngle(0), m_dN(0), 
    m_pElement(0),
    m_pGenElements(smf.elements())
{
  m_pElemBody = bodyBucket;
  m_pCommonBuckets = UAL_SXF_ElemBucketRegistry::getInstance(out);
}

// Destructor
UAL_SXF_Element::~UAL_SXF_Element()
{  
  if(m_pElemBody) { delete m_pElemBody; }
  if(m_pElement)  { delete m_pElement;  }
}

// Create a SMF lattice element.
int UAL_SXF_Element::openObject(const char* elementName, const char* type)
{
  m_pElement = new PacLattElement();
  m_pElement->name(elementName);
  m_pElement->key(string_to_key(m_sType));
  return SXF_TRUE;
}

// Add geometric attributes.
void UAL_SXF_Element::update()
{
  if(! m_pElement) return;

  if(m_dL) {
    PacElemLength length;
    length.l(m_dL);
    m_pElement->add(length);
  }
  if(m_dN) {
    PacElemComplexity complexity;
    complexity.n() = m_dN;
    m_pElement->add(complexity);
  }

}

// Clean up all temporary data.
void UAL_SXF_Element::close()
{
  if(m_pElement) { delete m_pElement;   m_pElement = 0;}
  m_dL = 0; 
  m_dAt = 0;
  m_dHAngle = 0;
  m_dN = 0;  
}

// Create a SMF generic element.
void UAL_SXF_Element::setDesign(const char* name)
{
  if(!m_pElement) return;

  PacGenElements::iterator it = m_pGenElements->find(name);
  if(it != m_pGenElements->end()){
    *m_pElement = (*it);
  }
  else{
    int key = string_to_key(m_sType);
    PacGenElement element(name, key);
    *m_pElement = element;
  }
}

// Set an element length.
void UAL_SXF_Element::setLength(double l)
{
  m_dL = l;
}

// Set a longitudinal position of the node with 
// respect to the beginning of the sequence.
void UAL_SXF_Element::setAt(double at)
{
  m_dAt = at;
}

// Set a horizontal angle.
void UAL_SXF_Element::setHAngle(double ha)
{
  m_dHAngle = ha;
}

// Set N.
void UAL_SXF_Element::setN(double n)
{
  m_dN = n;
}

// Get an element length
double UAL_SXF_Element::getLength() const
{
  return m_dL;
}

// Get a longitudinal position of the node with 
// respect to the beginning of the sequence
double  UAL_SXF_Element::getAt() const
{
  return m_dAt;
}

// Get a horizontal angle
double UAL_SXF_Element::getHAngle() const
{
  return m_dHAngle;
}

// Get N
double UAL_SXF_Element::getN() const
{
  return m_dN;
}

// Return a lattice element
PacLattElement* UAL_SXF_Element::getLattElement()
{
  return m_pElement;
}

// Add a bucket to the element.
void UAL_SXF_Element::addBucket(SXF::ElemBucket* bucket) 
{
  if(!m_pElement) return;

  UAL_SXF_ElemBucket* sxfBucket = (UAL_SXF_ElemBucket*) bucket;

  int i, size;
  PacElemBucket* smfBucket;
  
  // Get entry
  size = sxfBucket->getEntrySize();
  for(i = 0; i < size; i++) {
    smfBucket = sxfBucket->getEntryBucket(i);
    if(smfBucket) { m_pElement->setFront()->add(*smfBucket); }
  }

  // Get body
  size = sxfBucket->getBodySize();
  for(i = 0; i < size; i++) {
    smfBucket = sxfBucket->getBodyBucket(i);
    if(smfBucket) { m_pElement->setBody()->add(*smfBucket); }
  }

  // Get exit
  size = sxfBucket->getExitSize();
  for(i = 0; i < size; i++) {
    smfBucket = sxfBucket->getExitBucket(i);
    if(smfBucket) { m_pElement->setEnd()->add(*smfBucket); }
  }

}

// Map strings to the SMF keys.
int UAL_SXF_Element::string_to_key(const char* str) const
{
  SXF_Key* index = UAL_SXF_Perfect_Hash::smf_elements_gperf(str, strlen(str));

  if(index) { return index->number; }
  else return -1;
}

// Write an element into an output stream.
void UAL_SXF_Element::write(ostream& out, const PacLattElement& element, 
			    double at, const string& tab) 
{
  writeHeader(out, element, at, tab);
  writeBody(out, element, tab);
  writeCommonBuckets(out, element, tab);

  out << tab << "};" << endl;
}

// Write an element header.
void UAL_SXF_Element::writeHeader(ostream& out, 
				  const PacLattElement& element,
				  double at, const string& tab)
{
  double length = element.getLength();
  double n = element.getN();

  out << tab << element.name() << " " << m_sType << " {";

  if(element.genElement().name().length()) 
    out << " tag = " << element.genElement().name();
  if(at)      { out << " at = " << at ; } 
  if(length)  { out << " l = "  << length ; } 
  if(n)       { out << " n = "  << n ; } 
  out << endl;
}

// Write an element body.
void UAL_SXF_Element::writeBody(ostream& out, 
				const PacLattElement& element, 
				const string& tab)
{
  UAL_SXF_ElemBucket* writer = (UAL_SXF_ElemBucket*) m_pElemBody;
  if(writer) writer->write(out, element, tab + "  ");

}

// Write an element common buckets
void UAL_SXF_Element::writeCommonBuckets(ostream& out, 
				   const PacLattElement& element, 
				   const string& tab)
{
  UAL_SXF_ElemBucketRegistry* writer = 
    (UAL_SXF_ElemBucketRegistry*) m_pCommonBuckets;
  if(writer) writer->write(out, element, tab + "  ");
}


