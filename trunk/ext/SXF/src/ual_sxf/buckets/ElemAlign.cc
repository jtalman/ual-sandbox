#include "ual_sxf/buckets/ElemAlign.hh"

// Constructor.
UAL_SXF_ElemAlign::UAL_SXF_ElemAlign(SXF::OStream& out)
  : UAL_SXF_ElemBucket(out, "align", new SXF::ElemAlignHash())
{ 
  m_iIndex = -1;  

  m_iOffsetStatus = 0;
  m_iRotationStatus = 0;

  m_pOffset = new PacElemOffset();
  m_pRotation = new PacElemRotation();
}

// Destructor.
UAL_SXF_ElemAlign::~UAL_SXF_ElemAlign()
{
  if(m_pOffset) delete m_pOffset;
  if(m_pRotation) delete m_pRotation;

}

// Zero bucket.
void UAL_SXF_ElemAlign::close()
{
  m_pOffset->zero();
  m_pRotation->zero(); 

  m_iOffsetStatus = 0;
  m_iRotationStatus = 0;

  m_iIndex = -1;

}

// Check the attribute key. 
int UAL_SXF_ElemAlign::openArray()
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
void UAL_SXF_ElemAlign::closeArray()
{
}

// Do nothing. Return SXF_TRUE.
int UAL_SXF_ElemAlign::openHash()
{
  return SXF_TRUE;
}

// Do nothing. 
void UAL_SXF_ElemAlign::closeHash()
{
}

// Set the value at the current index in the array attribute
// and increment this index.
int UAL_SXF_ElemAlign::setArrayValue(double v)
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
int UAL_SXF_ElemAlign::setHashValue(double v, int index)
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
int UAL_SXF_ElemAlign::getBodySize() const
{
  return 2;
}

// Get the SMF body bucket selected by the given index (0 or 1).
// Return 0, if the bucket is not defined.
PacElemBucket* UAL_SXF_ElemAlign::getBodyBucket(int index) 
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
void UAL_SXF_ElemAlign::write(ostream& out, 
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

