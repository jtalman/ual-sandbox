#include "UAL/SXF/Error.hh"

// Constructor.
UAL::SXFError::SXFError(SXF::OStream& out, const char* type,
			     SXF::ElemBucket* bodyBucket, PacSmf& smf)
  : UAL::SXFElement(out, type, bodyBucket, smf)
{
}


// Print an error message and return SXF_FALSE.
int UAL::SXFError::openObject(const char* n, const char* type)
{
  m_refOStream.cfe_error() 
    << "\n*** UAL/SXF Error: " << n 
    << " has a wrong type (" << type << ")" << endl;
  return SXF_FALSE;
}
