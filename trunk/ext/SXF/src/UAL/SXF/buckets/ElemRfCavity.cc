#include "UAL/SXF/buckets/ElemRfCavity.hh"

// Constructor.
UAL::SXFElemRfCavity::SXFElemRfCavity(SXF::OStream& out)
  : UAL::SXFElemBucket(out, "body", new SXF::ElemRfCavityHash())
{
  m_iRfStatus = 0;
  m_pRfCavity = new PacElemRfCavity(0);
}

// Destructor.
UAL::SXFElemRfCavity::~SXFElemRfCavity()
{
  if(m_pRfCavity) delete m_pRfCavity;
}

// Zero bucket.
void UAL::SXFElemRfCavity::close()
{
  m_pRfCavity->order(-1);
  m_pRfCavity->order(0);
  m_iRfStatus = 0;
}

// Check the scalar attribute key (VOLT, LAG, HARMON) and set its value. 
// Skip the SHUNT and TFILL attribute.
int UAL::SXFElemRfCavity::setScalarValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemRfCavityHash::VOLT :
    m_pRfCavity->volt(0) = v;
    m_iRfStatus++;
    break;
  case SXF::ElemRfCavityHash::HARMON :
    m_pRfCavity->harmon(0) = v;
    m_iRfStatus++;
    break;
  case SXF::ElemRfCavityHash::LAG :
    m_pRfCavity->lag(0) = v;
    m_iRfStatus++;
    break;
  case SXF::ElemRfCavityHash::SHUNT :
    break;
  case SXF::ElemRfCavityHash::TFILL :
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return a number of SMF body buckets.
int UAL::SXFElemRfCavity::getBodySize() const
{
  return 1;
}

// Get the SMF rf cavity bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemRfCavity::getBodyBucket(int index) 
{
  if(index == 0)
    if(m_iRfStatus) return m_pRfCavity;
  return 0;
}

// Write data
void UAL::SXFElemRfCavity::write(ostream& out, const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = element.getBody();  
  if(!attributes) return;

  // Check rf bucket
  PacElemAttributes::iterator it = attributes->find(pacRfCavity.key());
  if(!(it != attributes->end())) return;

  *m_pRfCavity = (*it);
  int order = m_pRfCavity->order();
  double value = 0.0;

  if(order < 0) return;

  // Print it
  out << tab << m_sType << " = { ";

  if((value = m_pRfCavity->volt(0)) != 0)   out << "volt =  "   << value << " ";
  if((value = m_pRfCavity->harmon(0)) != 0) out << "harmon =  " << value << " ";
  if((value = m_pRfCavity->lag(0)) != 0)    out << "lag =  "    << value << " ";

  out << endl << tab << "}" << endl;
}


