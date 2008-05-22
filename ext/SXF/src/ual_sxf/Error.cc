#include "ual_sxf/Error.hh"

// Constructor.
UAL_SXF_Error::UAL_SXF_Error(SXF::OStream& out, const char* type,
			     SXF::ElemBucket* bodyBucket, PacSmf& smf)
  : UAL_SXF_Element(out, type, bodyBucket, smf)
{
}


// Print an error message and return SXF_FALSE.
int UAL_SXF_Error::openObject(const char* n, const char* type)
{
  m_refOStream.cfe_error() 
    << "\n*** UAL/SXF Error: " << n 
    << " has a wrong type (" << type << ")" << endl;
  return SXF_FALSE;
}
