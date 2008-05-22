#include "ual_sxf/buckets/ElemAperture.hh"

// Constructor
UAL_SXF_ElemAperture::UAL_SXF_ElemAperture(SXF::OStream& out)
  : UAL_SXF_ElemBucket(out, "aperture", new SXF::ElemApertureHash())
{
  m_pAperture = new PacElemAperture();
} 
 
// Destructor.
UAL_SXF_ElemAperture::~UAL_SXF_ElemAperture()
{
  if(m_pAperture) delete m_pAperture;
}
  

// Zero bucket.
void UAL_SXF_ElemAperture::close()
{
  m_pAperture->zero();
}  

// Check the scalar attribute key (XSIZE, YSIZE, SHAPE) and set its value. 
// Return SXF_TRUE or SXF_FALSE.
int UAL_SXF_ElemAperture::setScalarValue(double v)
{
  int result = SXF_TRUE;

  switch (m_iAttribID){
  case SXF::ElemApertureHash::XSIZE :
    m_pAperture->xsize() = v;
    break;
  case SXF::ElemApertureHash::YSIZE :
    m_pAperture->ysize() = v;
    break;
  case SXF::ElemApertureHash::SHAPE :
    m_pAperture->shape() = v;
    break;		      
  default:
    result = SXF_FALSE;
    break;
  };

  return result;
}

// Return 1 because the SXF Aperture body attributes are 
// represented by one SMF body bucket: Aperture.
int UAL_SXF_ElemAperture::getBodySize() const
{
  return 1;
}

// Get the SMF body bucket selected by the given index (0).
// Return 0, if the bucket is not defined.
PacElemBucket* UAL_SXF_ElemAperture::getBodyBucket(int index) 
{
  if(index) return 0;
  return m_pAperture;
}

// Write data.
void UAL_SXF_ElemAperture::write(ostream& out, const PacLattElement& element, const string& tab)
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

  if((v = (*it)[PAC_APERTURE_SHAPE]) != 0) out <<"shape = " << v <<endl;
  if((v = (*it)[PAC_APERTURE_XSIZE]) != 0) out <<tab<<tab<<tab<<tab<<" xsize = " << v <<endl;
  if((v = (*it)[PAC_APERTURE_YSIZE]) != 0) out <<tab<<tab<<tab<<tab<< " ysize = " << v <<endl;
 
  out << "}" << endl;
  
}

