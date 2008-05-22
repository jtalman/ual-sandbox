#include "echo_sxf/EchoAcceleratorReader.hh"
#include "echo_sxf/EchoNodeRegistry.hh"

// Constructor.
SXF::EchoAcceleratorReader::EchoAcceleratorReader(SXF::OStream& out)
  : SXF::AcceleratorReader(out)
{
  m_pNodeRegistry = SXF::EchoNodeRegistry::getInstance(out);
}
