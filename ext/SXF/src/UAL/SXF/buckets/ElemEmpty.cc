#include "UAL/SXF/buckets/ElemEmpty.hh"

// Constructor.
UAL::SXFElemEmpty::SXFElemEmpty(SXF::OStream& out, const char* type)
  : UAL::SXFElemBucket(out, type, new SXF::ElemEmptyHash())
{
}

// Ignore unsupported array attributes. Return SXF_TRUE.
int UAL::SXFElemEmpty::openArray()
{
  return SXF_TRUE;
}

// Return SXF_TRUE.
int UAL::SXFElemEmpty::setArrayValue(double)
{
  return SXF_TRUE;
}

// Ignore unsupported scalar attributes. Return SXF_TRUE.
int UAL::SXFElemEmpty::openAttribute(const char* name)
{
  SXF::ElemBucket::openAttribute(name);
  return SXF_TRUE;
}

// Ignore unsupported scalar attributes. Return SXF_TRUE.
int UAL::SXFElemEmpty::setScalarValue(double)
{
  return SXF_TRUE;
}
