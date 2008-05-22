#include "ual_sxf/buckets/ElemCollimator.hh"

// Constructor
UAL_SXF_ElemCollimator::UAL_SXF_ElemCollimator(SXF::OStream& out, 
					       const char* type,
					       const char shape)
  : UAL_SXF_ElemBucket(out, type, new SXF::ElemCollimatorHash())
{
  m_cShape = shape;
  m_pAperture = new PacElemAperture();
}

// Destructor.
UAL_SXF_ElemCollimator::~UAL_SXF_ElemCollimator()
{
  if(m_pAperture) delete m_pAperture;
}

// Define the aperture shape.
void UAL_SXF_ElemCollimator::update()
{
  switch (m_cShape) {
  case 'e' :
    m_pAperture->shape() = 1.0;
    break;
  case 'r' :
    m_pAperture->shape() = 2.0;
    break;
  default:
    m_pAperture->shape() = 1.0; 
    break;    
  };
}

// Zero bucket.
void UAL_SXF_ElemCollimator::close()
{
  m_pAperture->zero();
}

// Check the scalar attribute key (XSIZE, YSIZE) and set its value. 
// Return SXF_TRUE or SXF_FALSE.
int UAL_SXF_ElemCollimator::setScalarValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemCollimatorHash::XSIZE :
    m_pAperture->xsize() = v;
    break;
  case SXF::ElemCollimatorHash::YSIZE :
    m_pAperture->ysize() = v;
    break;
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return 1 because the SXF collimator body attributes are 
// represented by one SMF body bucket: Aperture.
int UAL_SXF_ElemCollimator::getBodySize() const
{
  return 1;
}

// Get the SMF body bucket selected by the given index (0).
// Return 0, if the bucket is not defined.
PacElemBucket* UAL_SXF_ElemCollimator::getBodyBucket(int index) 
{
  if(index) return 0;
  return m_pAperture;
}

// Write data.
void UAL_SXF_ElemCollimator::write(ostream& out, const PacLattElement& element, const string& tab)
{
  // Get body attributes
  PacElemAttributes* attributes = element.getBody();  
  if(!attributes) return;

  // Check aperture bucket
  PacElemAttributes::iterator it = attributes->find(pacAperture.key());
  if(!(it != attributes->end())) return;

  out << tab << m_sType << " = { ";

  // Get attribute
  double v;

  if((v = (*it)[PAC_APERTURE_XSIZE]) != 0) out << "xsize = " << v << " ";
  if((v = (*it)[PAC_APERTURE_YSIZE]) != 0) out << "ysize = " << v << " ";

  out << "}" << endl;
  
}

