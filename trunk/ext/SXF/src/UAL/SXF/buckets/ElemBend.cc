#include "UAL/SXF/buckets/ElemBend.hh"

int UAL::SXFElemBend::s_iMaxOrder = UAL_SXF_MAX_ORDER;

// Constructor.
UAL::SXFElemBend::SXFElemBend(SXF::OStream& out)
  : UAL::SXFElemBucket(out, "body", new SXF::ElemBendHash())
{
  // Deviation flag
  m_iIsDeviation = 0;

  // Bend data (entry, exit, end)
  for(int i=0; i < 3; i++){
    m_pBend[i] = new PacElemBend();
    m_iBendStatus[i] =  0;
  }

  // Multipole data
  m_pMultipole  = new PacElemMultipole(s_iMaxOrder);
  m_iOrderKL    = -1;
  m_iOrderKTL   = -1;

  makeMltFactors();

}

// Destructor.
UAL::SXFElemBend::~SXFElemBend()
{
  // Bend data
  for(int i=0; i < 3; i++){
    if(m_pBend[i]) delete m_pBend[i];
  }
  // Multipole data
  if(m_pMultipole) delete m_pMultipole;

  if(m_aMltFactor) delete [] m_aMltFactor;
}

int UAL::SXFElemBend::isDeviation() const 
{
  return m_iIsDeviation;
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "body.dev".
int UAL::SXFElemBend::openObject(const char* type)
{
  if(!strcmp(type, "body.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Update bucket: Define the max order of the multipole harmonics.
void UAL::SXFElemBend::update()
{
  int i, order = m_iOrderKL > m_iOrderKTL ? m_iOrderKL : m_iOrderKTL;

  for(i = m_iOrderKL  + 1; i < order; i++) { m_pMultipole->kl(i)  = 0.0;}
  for(i = m_iOrderKTL + 1; i < order; i++) { m_pMultipole->ktl(i) = 0.0;}

  m_pMultipole->order(order);
}

// Zero bucket.
void UAL::SXFElemBend::close()
{
  // Bend data (entry, body, end)
  for(int i=0; i < 3; i++){ 
    m_pBend[i]->zero();
    m_iBendStatus[i] =  0;
  }

  // Multipole data
  m_pMultipole->order(-1);
  m_pMultipole->order(s_iMaxOrder);
  m_iOrderKL    = -1;
  m_iOrderKTL   = -1;
}

// Check the array attribute key.
// Return SXF_TRUE or SXF_FALSE. 
int UAL::SXFElemBend::openArray()
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

// Do nothing.
void UAL::SXFElemBend::closeArray()
{
}

// Check the array attribute key (KL or KLS). 
// Return SXF_TRUE or SXF_FALSE.
int UAL::SXFElemBend::openHash()
{
  return openArray();
}

// Do nothing. 
void UAL::SXFElemBend::closeHash()
{
}

// Check the scalar attribute key (FINT, E1, E2) and set its value. 
// Skip the HGAP attribute.
int UAL::SXFElemBend::setScalarValue(double v)
{
  int result = SXF_TRUE;

  if(!v) return result;

  switch (m_iAttribID){
  case SXF::ElemBendHash::FINT :
    m_pBend[1]->fint() = v;
    m_iBendStatus[1]++;
    break;
  case SXF::ElemBendHash::E1 :
    m_pBend[0]->angle() = v;
    m_iBendStatus[0]++;
    break;
  case SXF::ElemBendHash::E2 :
    m_pBend[2]->angle() = v;
    m_iBendStatus[2]++;
    break;
  case SXF::ElemBendHash::HGAP :
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Set the value at the current index in the array 
// attribute (KL or KLS) and increment this index. 
// Define the SMF bend angle if it is KL[0] and is
// not a deviation.
int UAL::SXFElemBend::setArrayValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemBendHash::KL :
    ++m_iOrderKL;
    if(m_iOrderKL == 0 && m_iIsDeviation == 0){
      m_pBend[1]->angle() = v;
      m_iBendStatus[1]++;
      m_pMultipole->kl(m_iOrderKL) = 0.0;
    }
    else{
      m_pMultipole->kl(m_iOrderKL) = v/m_aMltFactor[m_iOrderKL];
    }
    break;
  case SXF::ElemBendHash::KLS :
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
// Define the SMF bend angle if it is KL[0] and is not a deviation.
int UAL::SXFElemBend::setHashValue(double v, int index)
{
  int result = SXF_TRUE;

  if(index < 0) {
    m_refOStream.cfe_error() 
      << "\n*** UAL/SXF Error : index(" << index << ") < 0" << endl;
    return SXF_FALSE;
  }

  switch (m_iAttribID){
  case SXF::ElemBendHash::KL :
    m_iOrderKL = m_iOrderKL > index ? m_iOrderKL : index;   
    if(index == 0 && m_iIsDeviation == 0){
      m_pBend[1]->angle() = v;
      m_iBendStatus[1]++;
      m_pMultipole->kl(index) = 0.0;
    }
    else {
      m_pMultipole->kl(index) = v/m_aMltFactor[index];
    }
    break;
  case SXF::ElemBendHash::KLS :
    m_iOrderKTL = m_iOrderKTL > index ? m_iOrderKTL : index;
    m_pMultipole->ktl(index) = v/m_aMltFactor[index];
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return 1 because the SXF E1 attribute is represented
// by the SMF entry bucket attribute ANGLE.
int UAL::SXFElemBend::getEntrySize() const
{
  return 1;
}

// Return 2 because the SXF bend body attributes are represented 
// by two SMF body buckets: Bend and Multipole.
int UAL::SXFElemBend::getBodySize() const
{
  return 2;
}

// Return 1 because the SXF E2 attribute is represented
// by the SMF exit bucket attribute ANGLE.
int UAL::SXFElemBend::getExitSize() const
{
  return 1;
}

// Get the SMF entry bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemBend::getEntryBucket(int index) 
{
  if(index == 0) 
    if(m_iBendStatus[0]) return m_pBend[0];
  return 0;
}

// Get the SMF exit bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemBend::getExitBucket(int index) 
{
  if(index == 0) 
    if(m_iBendStatus[2]) return m_pBend[2];
  return 0;
}

// Get the SMF body bucket selected by the given index (0 or 1).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemBend::getBodyBucket(int index) 
{
  if(index == 0) { // Bend
    if(m_iBendStatus[1]) return m_pBend[1];
  }
  if(index == 1) { // Multipole
    if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
    return m_pMultipole;
  }

  return 0;
}

void UAL::SXFElemBend::write(ostream& out, 
const PacLattElement& element, const string& tab)
{

  // Get attributes
  PacElemAttributes*  front = element.getFront();  
  PacElemAttributes*  body  = element.getBody(); 
  PacElemAttributes*  end   = element.getEnd();  

  double v, hangle = 0.0;
  PacElemAttributes::iterator it; 


  // Print it
  out << tab << m_sType << " = { ";

  // Bend data

  // fint + angle
  if(body){
    it = body->find(pacBend.key());
    if(it != body->end()) {
      v = (*it)[PAC_BEND_FINT];
      hangle = (*it)[PAC_BEND_ANGLE];
      if(v) out << "fint = " << v << " " ;
    }
  }

  // e1
  if(front){
    it = front->find(pacBend.key());
    if(it != front->end()) {
      v = (*it)[PAC_BEND_ANGLE];
      if(v) out << "e1 = " << v << " ";
    }
  }  

  // e2
  if(end){
    it = end->find(pacBend.key());
    if(it != end->end()) {
      v = (*it)[PAC_BEND_ANGLE];
      if(v) out << "e2 = " << v << " ";
    }
  }

  if(hangle){
    out << "kl = [ " << hangle << "]";
  }
  out << "}" << endl;

  // Multipole data

  if(body){

    it = body->find(pacMultipole.key());
    if(it == body->end()) return;

    *m_pMultipole = (*it);
    int i, order = m_pMultipole->order();

    if(order < 0) return;

    int isEmpty = 1;

    for(i=0; i <= order; i++){
      if((m_pMultipole->kl(i) != 0.0) || (m_pMultipole->ktl(i) != 0.0)) {
	isEmpty = 0;
	break;
      }
    }

    if(isEmpty) return;

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

    out << endl << tab;

    out << "}" << endl;
  }

}

// Make an array of SMF-to-SIF multipole factors.  
void UAL::SXFElemBend::makeMltFactors()
{ 
  m_aMltFactor = new double[s_iMaxOrder+1];
  m_aMltFactor[0] = 1;
  for(int i=1; i <= s_iMaxOrder; i++){
    m_aMltFactor[i] = m_aMltFactor[i-1]*i;
  }
}
