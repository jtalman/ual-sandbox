#include "UAL/SXF/buckets/ElemMltBody.hh"

// Constructor.
UAL::SXFElemMltBody::SXFElemMltBody(SXF::OStream& out)
  : UAL::SXFElemMultipole(out, "body")
{ 
  m_pBend = new PacElemBend();
  m_iBendStatus =  0;    
}

// Destructor
UAL::SXFElemMltBody::~SXFElemMltBody()
{
  if(m_pBend) delete m_pBend;
}

// Zero bucket.
void UAL::SXFElemMltBody::close()
{
  m_pBend->zero();
  m_iBendStatus =  0; 

  UAL::SXFElemMultipole::close();
}

// Set the value at the current index in the array 
// attribute (KL or KLS) and increment this index. 
// Skip the attribute if it is KL[0] and is not a deviation.
int UAL::SXFElemMltBody::setArrayValue(double v)
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
int UAL::SXFElemMltBody::setHashValue(double v, int index)
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
int UAL::SXFElemMltBody::getBodySize() const
{
  return 2;
}

// Get the SMF body bucket selected by the given index (0 or 1).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemMltBody::getBodyBucket(int index) 
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

void UAL::SXFElemMltBody::write(ostream& out, const PacLattElement& element, const string& tab)
{

  // Write design mlt attributes

  PacElemMultipole genMultipole = getDesignMultipole(element);  
  if(genMultipole.order() > -1 ) writeMultipole(out, genMultipole, 0, tab);

  // Get deviation mlt attributes

  PacElemMultipole devMultipole = getTotalMultipole(element);
  devMultipole -= genMultipole;
  if(devMultipole.order() > -1 ) writeMultipole(out, devMultipole, 1, tab);  

  return;

}

// PacElemAttributes* UAL::SXFElemMltBody::getAttributes(const PacLattElement& element)
// {
//  return element.getBody();
// }

// Return exit design multipole
PacElemMultipole UAL::SXFElemMltBody::getDesignMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemPart* part = element.genElement().getBody();
  if(!part) return result;

  PacElemAttributes& attributes = part->attributes();

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes.find(pacMultipole.key());
  if(!(it != attributes.end())) return result;

  return static_cast<PacElemMultipole>( (*it));  
}

// Return exit design multipole
PacElemMultipole UAL::SXFElemMltBody::getTotalMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemAttributes* attributes = element.getBody();
  if(!attributes) return result;

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes->find(pacMultipole.key());
  if(!(it != attributes->end())) return result;

  return (*it);  
}




