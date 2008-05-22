#include "ual_sxf/buckets/ElemError.hh"

// Constructor.
UAL_SXF_ElemError::UAL_SXF_ElemError(SXF::OStream& out)
  : UAL_SXF_ElemBucket(out, "error", new SXF::ElemEmptyHash())
{
}

// Print an error message.
int UAL_SXF_ElemError::openObject(const char* type)
{
  m_refOStream.cfe_error() 
    << "\n*** UAL/SXF Error: " << type
    <<  " is a wrong element bucket type" << endl;
  return SXF_FALSE;
}
