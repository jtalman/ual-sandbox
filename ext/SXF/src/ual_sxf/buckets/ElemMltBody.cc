#include "ual_sxf/buckets/ElemMltBody.hh"

// Constructor.
UAL_SXF_ElemMltBody::UAL_SXF_ElemMltBody(SXF::OStream& out)
  : UAL_SXF_ElemMultipole(out, "body")
{ 
  m_pBend = new PacElemBend();
  m_iBendStatus =  0;    
}

// Destructor
UAL_SXF_ElemMltBody::~UAL_SXF_ElemMltBody()
{
  if(m_pBend) delete m_pBend;
}

// Zero bucket.
void UAL_SXF_ElemMltBody::close()
{
  m_pBend->zero();
  m_iBendStatus =  0; 

  UAL_SXF_ElemMultipole::close();
}

// Set the value at the current index in the array 
// attribute (KL or KLS) and increment this index. 
// Skip the attribute if it is KL[0] and is not a deviation.
int UAL_SXF_ElemMltBody::setArrayValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemMultipoleHash::KL :
    ++m_iOrderKL;
    if(m_iOrderKL == 0 && m_iIsDeviation == 0){
      m_pBend->angle() = v;
      m_iBendStatus++;
      m_pMultipole->kl(m_iOrderKL) = 0.0;
    }
    else {
      m_pMultipole->kl(m_iOrderKL) = v/m_aMltFactor[m_iOrderKL];
    }
    break;
  case SXF::ElemMultipoleHash::KLS :
    ++m_iOrderKTL;
    m_pMultipole->ktl(m_iOrderKTL) = v/m_aMltFactor[m_iOrderKTL];
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Set the value at given index in the array attribute (KL or KLS).
// Skip the attribute if it is KL[0] and is not a deviation.
int UAL_SXF_ElemMltBody::setHashValue(double v, int index)
{
  int result = SXF_TRUE;

  if(index < 0) {
    m_refOStream.cfe_error() 
      << "\n*** UAL/SXF Error : index(" << index << ") < 0" << endl;
    return SXF_FALSE;
  }

  switch (m_iAttribID){
  case SXF::ElemMultipoleHash::KL :
    m_iOrderKL = m_iOrderKL > index ? m_iOrderKL : index;   
    if(index == 0 && m_iIsDeviation == 0){
      m_pBend->angle() = v;
      m_iBendStatus++;
      m_pMultipole->kl(index) = 0.0;
    }
    else {
      m_pMultipole->kl(index) = v/m_aMltFactor[index];
    }
    break;
  case SXF::ElemMultipoleHash::KLS :
    m_iOrderKTL = m_iOrderKTL > index ? m_iOrderKTL : index;
    m_pMultipole->ktl(index) = v/m_aMltFactor[index];
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return a number of SMF body buckets.
int UAL_SXF_ElemMltBody::getBodySize() const
{
  return 2;
}

// Get the SMF body bucket selected by the given index (0 or 1).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL_SXF_ElemMltBody::getBodyBucket(int index) 
{
  if(index == 0) { // Bend
    if(m_iBendStatus) return m_pBend;
  }
  if(index == 1) { // Multipole
    if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
    return m_pMultipole;
  }

  return 0;
}

void UAL_SXF_ElemMltBody::write(ostream& out, const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = getAttributes(element);  
  if(!attributes) return;

  PacElemAttributes::iterator it;

  // Check bend bucket

  it = attributes->find(pacBend.key());
  if(it != attributes->end()) {
    double hangle = (*it)[PAC_BEND_ANGLE];
    if(hangle){
      out << tab << m_sType << " = { ";
      out << "kl = [ " << hangle << "]";
      out << "}" << endl;
    } 
  }

  // Check multipole bucket
  it = attributes->find(pacMultipole.key());
  if(!(it != attributes->end())) return;

  *m_pMultipole = (*it);
  int i, order = m_pMultipole->order();

  if(order < 0) return;

  // Print it
  out << tab << m_sType << ".dev = { ";

  if(order > 5) out << endl << tab << "    ";
  out << "kl = [ ";
  for(i=0; i <= order; i++)  
    out << m_pMultipole->kl(i)*m_aMltFactor[i] << " ";
  out << "] ";

  if(order > 5) out << endl << tab << "    ";

  out << "kls = [ ";
  for(i=0; i <= order; i++)  
    out << m_pMultipole->ktl(i)*m_aMltFactor[i] << " ";
  out << "] "; 

  out << endl << tab << "}" << endl;
}

PacElemAttributes* UAL_SXF_ElemMltBody::getAttributes(const PacLattElement& element)
{
  return element.getBody();
}
