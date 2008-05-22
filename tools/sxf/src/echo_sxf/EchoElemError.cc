#include "echo_sxf/EchoElemError.hh"

// Constructor.
SXF::EchoElemError::EchoElemError(SXF::OStream& out)
  : SXF::EchoElemBucket(out, "error", new SXF::ElemEmptyHash())
{
}

// Print an error message
int SXF::EchoElemError::openObject(const char* type)
{
  m_refOStream.cfe_error() 
    << "\n*** SXF ECHO Error: " << type
    <<  " is a wrong element bucket type" << endl;
  return SXF_FALSE;
}
