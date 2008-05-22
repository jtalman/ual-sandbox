#include "UAL/SXF/buckets/ElemMultipole.hh"

int UAL::SXFElemMultipole::s_iMaxOrder = UAL_SXF_MAX_ORDER;

// Constructor.
UAL::SXFElemMultipole::SXFElemMultipole(SXF::OStream& out, const char* type)
  : UAL::SXFElemBucket(out, type, new SXF::ElemMultipoleHash())
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
UAL::SXFElemMultipole::~SXFElemMultipole()
{
  if(m_pMultipole) delete m_pMultipole;
  if(m_aMltFactor) delete [] m_aMltFactor;
}

int UAL::SXFElemMultipole::isDeviation() const
{
  return m_iIsDeviation;
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "body.dev".
int UAL::SXFElemMultipole::openObject(const char* type)
{
  if(!strcmp(type, "body.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Update bucket: Define the max order of the multipole harmonics.
void UAL::SXFElemMultipole::update()
{
  int i, order = m_iOrderKL > m_iOrderKTL ? m_iOrderKL : m_iOrderKTL;

  for(i = m_iOrderKL  + 1; i < order; i++) { m_pMultipole->kl(i)  = 0.0;}
  for(i = m_iOrderKTL + 1; i < order; i++) { m_pMultipole->ktl(i) = 0.0;}

  m_pMultipole->order(order);
}

// Zero bucket.
void UAL::SXFElemMultipole::close()
{
  m_pMultipole->order(-1);
  m_pMultipole->order(s_iMaxOrder);
  m_iOrderKL  = -1;
  m_iOrderKTL = -1;
}

// Check the array attribute key.
// Return SXF_TRUE or SXF_FALSE. 
int UAL::SXFElemMultipole::openArray()
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
int UAL::SXFElemMultipole::openHash()
{
  return openArray();
}

// Skip the LRAD attribute.
int UAL::SXFElemMultipole::setScalarValue(double v)
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
int UAL::SXFElemMultipole::setArrayValue(double v)
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
int UAL::SXFElemMultipole::setHashValue(double v, int index)
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

void UAL::SXFElemMultipole::write(ostream& out, const PacLattElement& element, const string& tab)
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


void UAL::SXFElemMultipole::writeMultipole(ostream& out, PacElemMultipole& multipole, int isDev, const string& tab)
{
  int i;
  int order = multipole.order();

  if(order < 0) return;

  int isEmpty = 1;

  for(i=0; i <= order; i++){
    if((multipole.kl(i) != 0.0) || (multipole.ktl(i) != 0.0)) {
      isEmpty = 0;
      break;
    }
  }

  if(isEmpty) return;

  // Print it
  out << tab << m_sType ;

  if(isDev) out << ".dev";

  out << " = { ";

  if(order > 5) out << endl << tab << "    ";
  out << "kl = [ ";
  for(i=0; i <= order; i++)  
    out << multipole.kl(i)*m_aMltFactor[i] << " ";
  out << "] ";

  if(order > 5) out << endl << tab << "    ";

  out << "kls = [ ";
  for(i=0; i <= order; i++)  
    out << multipole.ktl(i)*m_aMltFactor[i] << " ";
  out << "] "; 

  out << endl << tab << "}" << endl;
}

// Make an array of SMF-to-SIF multipole factors. 
void UAL::SXFElemMultipole::makeMltFactors()
{ 
  m_aMltFactor = new double[s_iMaxOrder+1];
  m_aMltFactor[0] = 1;
  for(int i=1; i <= s_iMaxOrder; i++){
    m_aMltFactor[i] = m_aMltFactor[i-1]*i;
  }
}
