#include "ual_sxf/buckets/ElemMultipole.hh"

int UAL_SXF_ElemMultipole::s_iMaxOrder = UAL_SXF_MAX_ORDER;

// Constructor.
UAL_SXF_ElemMultipole::UAL_SXF_ElemMultipole(SXF::OStream& out, const char* type)
  : UAL_SXF_ElemBucket(out, type, new SXF::ElemMultipoleHash())
{  
  // Deviation flag
  m_iIsDeviation = 0;

  // Multipole data
  m_pMultipole = new PacElemMultipole(s_iMaxOrder);
  m_iOrderKL   = -1;
  m_iOrderKTL  = -1;

  makeMltFactors();
}

// Destructor.
UAL_SXF_ElemMultipole::~UAL_SXF_ElemMultipole()
{
  if(m_pMultipole) delete m_pMultipole;
  if(m_aMltFactor) delete [] m_aMltFactor;
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "body.dev".
int UAL_SXF_ElemMultipole::openObject(const char* type)
{
  if(!strcmp(type, "body.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Update bucket: Define the max order of the multipole harmonics.
void UAL_SXF_ElemMultipole::update()
{
  int i, order = m_iOrderKL > m_iOrderKTL ? m_iOrderKL : m_iOrderKTL;

  for(i = m_iOrderKL  + 1; i < order; i++) { m_pMultipole->kl(i)  = 0.0;}
  for(i = m_iOrderKTL + 1; i < order; i++) { m_pMultipole->ktl(i) = 0.0;}

  m_pMultipole->order(order);
}

// Zero bucket.
void UAL_SXF_ElemMultipole::close()
{
  m_pMultipole->order(-1);
  m_pMultipole->order(s_iMaxOrder);
  m_iOrderKL  = -1;
  m_iOrderKTL = -1;
}

// Check the array attribute key.
// Return SXF_TRUE or SXF_FALSE. 
int UAL_SXF_ElemMultipole::openArray()
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemBendHash::KL :
    break;
  case SXF::ElemBendHash::KLS :
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Check the array attribute key (KL or KLS). 
// Return SXF_TRUE or SXF_FALSE.
int UAL_SXF_ElemMultipole::openHash()
{
  return openArray();
}

// Skip the LRAD attribute.
int UAL_SXF_ElemMultipole::setScalarValue(double v)
{
  int result = SXF_TRUE;

  if(!v) return result;

  switch (m_iAttribID){
  case SXF::ElemMultipoleHash::LRAD :
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Set the value at the current index in the array 
// attribute (KL or KLS) and increment this index. 
int UAL_SXF_ElemMultipole::setArrayValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemMultipoleHash::KL :
    ++m_iOrderKL;
    m_pMultipole->kl(m_iOrderKL) = v/m_aMltFactor[m_iOrderKL];
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
int UAL_SXF_ElemMultipole::setHashValue(double v, int index)
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
    m_pMultipole->kl(index) = v/m_aMltFactor[m_iOrderKTL];
    break;
  case SXF::ElemMultipoleHash::KLS :
    m_iOrderKTL = m_iOrderKTL > index ? m_iOrderKTL : index;
    m_pMultipole->ktl(index) = v/m_aMltFactor[m_iOrderKTL];
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

void UAL_SXF_ElemMultipole::write(ostream& out, const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = getAttributes(element);  
  if(!attributes) return;

  // Check multipole bucket
  PacElemAttributes::iterator it = attributes->find(pacMultipole.key());
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

// Make an array of SMF-to-SIF multipole factors. 
void UAL_SXF_ElemMultipole::makeMltFactors()
{ 
  m_aMltFactor = new double[s_iMaxOrder+1];
  m_aMltFactor[0] = 1;
  for(int i=1; i <= s_iMaxOrder; i++){
    m_aMltFactor[i] = m_aMltFactor[i-1]*i;
  }
}
