#include "sxf/Sequence.hh"

SXF::Sequence::Sequence(SXF::OStream& out)
  : SXF::AcceleratorNode(out, "sequence")
{
}

// Return false
int SXF::Sequence::isElement() const
{
  return SXF_FALSE;
}


