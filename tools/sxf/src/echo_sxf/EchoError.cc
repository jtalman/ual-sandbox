#include "echo_sxf/EchoElemBucket.hh"
#include "echo_sxf/EchoError.hh"

// Constructor.
SXF::EchoError::EchoError(SXF::OStream& out, const char* type)
  : SXF::EchoElement(out, type, 0)
{
  m_pElemBody = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
}

// Print an error message and return SXF_FALSE.
int SXF::EchoError::openObject(const char* n, const char* type)
{
  m_refOStream.cfe_error() << "\n*** SXF ECHO Error: " << n 
			   << " has a wrong type (" << type << ")" << endl;
  return SXF_FALSE;
}
