#include "UAL/SXF/buckets/ElemError.hh"

// Constructor.
UAL::SXFElemError::SXFElemError(SXF::OStream& out)
  : UAL::SXFElemBucket(out, "error", new SXF::ElemEmptyHash())
{
}

// Print an error message.
int UAL::SXFElemError::openObject(const char* type)
{
  m_refOStream.cfe_error() 
    << "\n*** UAL/SXF Error: " << type
    <<  " is a wrong element bucket type" << endl;
  return SXF_FALSE;
}
