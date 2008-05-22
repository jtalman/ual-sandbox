#include "UAL/SXF/buckets/ElemAlign.hh"

// Constructor.
UAL::SXFElemAlign::SXFElemAlign(SXF::OStream& out)
  : UAL::SXFElemBucket(out, "align", new SXF::ElemAlignHash())
{ 
  // Deviation flag
  m_iIsDeviation = 0;

  m_iIndex = -1;  

  m_iOffsetStatus = 0;
  m_iRotationStatus = 0;

  m_pOffset = new PacElemOffset();
  m_pRotation = new PacElemRotation();
}

// Destructor.
UAL::SXFElemAlign::~SXFElemAlign()
{
  if(m_pOffset) delete m_pOffset;
  if(m_pRotation) delete m_pRotation;

}

int UAL::SXFElemAlign::isDeviation() const
{
  return m_iIsDeviation;
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "align.dev".
int UAL::SXFElemAlign::openObject(const char* type)
{
  if(!strcmp(type, "align.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Zero bucket.
void UAL::SXFElemAlign::close()
{
  m_pOffset->zero();
  m_pRotation->zero(); 

  m_iOffsetStatus = 0;
  m_iRotationStatus = 0;

  m_iIndex = -1;

}

// Check the attribute key. 
int UAL::SXFElemAlign::openArray()
{
  int result = SXF_TRUE;
  switch (m_iAttribID){
  case SXF::ElemAlignHash::AL :
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;  
}

// Do nothing.
void UAL::SXFElemAlign::closeArray()
{
}

// Do nothing. Return SXF_TRUE.
int UAL::SXFElemAlign::openHash()
{
  return SXF_TRUE;
}

// Do nothing. 
void UAL::SXFElemAlign::closeHash()
{
}

// Set the value at the current index in the array attribute
// and increment this index.
int UAL::SXFElemAlign::setArrayValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemAlignHash::AL :
    ++m_iIndex; if(v) {
      switch(m_iIndex) {
      case 0 :
	m_pOffset->dx() = v;
	m_iOffsetStatus++;
	break;
      case 1 :
	m_pOffset->dy() = v;
	m_iOffsetStatus++;
	break;
      case 2 :  
	m_pOffset->ds() = v;
	m_iOffsetStatus++;
	break; 
      case 3 :
	m_pRotation->dphi() = v;
	m_iRotationStatus++;
	break;
      case 4 :
	m_pRotation->dtheta() = v;
	m_iRotationStatus++;
	break;
      case 5 :  
	m_pRotation->tilt() = v;
	m_iRotationStatus++;
      break;   
      default:
	result = SXF_FALSE;
	break;   
      };
    }
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Set the value at given index in the array attribute.
int UAL::SXFElemAlign::setHashValue(double v, int index)
{
  int result = SXF_TRUE;
  if(!v) return result;

  if(index < 0) {
    m_refOStream.cfe_error() 
      << "\n*** UAL/SXF Error : index(" << index << ") < 0" << endl;
    return SXF_FALSE;
  }

  switch (m_iAttribID){
  case SXF::ElemAlignHash::AL :      
    switch(index) {
    case 0 :
      m_pOffset->dx() = v;
      m_iOffsetStatus++;
      break;
    case 1 :
      m_pOffset->dy() = v;
      m_iOffsetStatus++;
      break;
    case 2 :  
      m_pOffset->ds() = v;
      m_iOffsetStatus++;
      break; 
    case 3 :
      m_pRotation->dphi() = v;
      m_iRotationStatus++;
      break;
    case 4 :
      m_pRotation->dtheta() = v;
      m_iRotationStatus++;
      break;
    case 5 :  
      m_pRotation->tilt() = v;
      m_iRotationStatus++;
      break;   
    default:
      result = SXF_FALSE;
      break;   
    };
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return 2 because align attributes represented by 
// two SMF body buckets: Offset (0) and Rotation (1). 
int UAL::SXFElemAlign::getBodySize() const
{
  return 2;
}

// Get the SMF body bucket selected by the given index (0 or 1).
// Return 0, if the bucket is not defined.
PacElemBucket* UAL::SXFElemAlign::getBodyBucket(int index) 
{
  if(index == 0) { // Offset
    if(m_iOffsetStatus) return m_pOffset;
  }
  if(index == 1) { // Rotation
    if(m_iRotationStatus) return m_pRotation;
  }

  return 0;
}

// Write data.
void UAL::SXFElemAlign::write(ostream& out, 
const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = element.getBody();  
  if(!attributes) return;

  int offsetFlag = 0, rotationFlag = 0;
  PacElemAttributes::iterator it;

  // Check offset bucket
  it = attributes->find(pacOffset.key());
  if(it != attributes->end()){
    *m_pOffset = (*it);
    offsetFlag = 1;
  }

  it = attributes->find(pacRotation.key());
  if(it != attributes->end()){
    *m_pRotation = (*it);
    rotationFlag = 1;
  }

  if(offsetFlag + rotationFlag) {
    out << tab << m_sType << " = { al = [ ";
    out << m_pOffset->dx() << " " 
	<< m_pOffset->dy() << " " 
	<< m_pOffset->ds() << " ";
    if(rotationFlag) {
      out << m_pRotation->dphi() << " " 
	  << m_pRotation->dtheta() << " " 
	  << m_pRotation->tilt() << " ";
    }
    out << "]" << endl << tab << "}" << endl;
  }
    
}

