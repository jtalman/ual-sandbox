#include "UAL/SXF/buckets/ElemSolenoid.hh"

// Constructor.
UAL::SXFElemSolenoid::SXFElemSolenoid(SXF::OStream& out)
  : UAL::SXFElemBucket(out, "body", new SXF::ElemSolenoidHash())
{
  m_iSolenoidStatus = 0;
  m_pSolenoid = new PacElemSolenoid();
}

// Destructor.
UAL::SXFElemSolenoid::~SXFElemSolenoid()
{
  if(m_pSolenoid) delete m_pSolenoid;
}

// Zero bucket.
void UAL::SXFElemSolenoid::close()
{
  m_pSolenoid->zero();
  m_iSolenoidStatus = 0;
}

// Check the scalar attribute key (KS) and set its value. 
int UAL::SXFElemSolenoid::setScalarValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemSolenoidHash::KS :
    m_pSolenoid->ks(v);
    m_iSolenoidStatus++;
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return a number of SMF buckets.
int UAL::SXFElemSolenoid::getBodySize() const
{
  return 1;
}

// Get the SMF solenoid bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemSolenoid::getBodyBucket(int index) 
{
  if(index == 0)
    if(m_iSolenoidStatus) return m_pSolenoid;
  return 0;
}

// Write data.
void UAL::SXFElemSolenoid::write(ostream& out, const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = element.getBody();  
  if(!attributes) return;

  // Check solenoid bucket
  PacElemAttributes::iterator it = attributes->find(pacSolenoid.key());
  if(!(it != attributes->end())) return;

  // Get attribute
  double ks = (*it)[PAC_SOLENOID_KS];
  if(!ks) return;

  // Print it
  out << tab << m_sType << " = { ks = " << ks << "}" << endl;
  
}

