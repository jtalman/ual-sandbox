#include "ual_sxf/buckets/ElemEmpty.hh"

// Constructor.
UAL_SXF_ElemEmpty::UAL_SXF_ElemEmpty(SXF::OStream& out, const char* type)
  : UAL_SXF_ElemBucket(out, type, new SXF::ElemEmptyHash())
{
}

// Ignore unsupported array attributes. Return SXF_TRUE.
int UAL_SXF_ElemEmpty::openArray()
{
  return SXF_TRUE;
}

// Return SXF_TRUE.
int UAL_SXF_ElemEmpty::setArrayValue(double)
{
  return SXF_TRUE;
}

// Ignore unsupported scalar attributes. Return SXF_TRUE.
int UAL_SXF_ElemEmpty::openAttribute(const char* name)
{
  SXF::ElemBucket::openAttribute(name);
  return SXF_TRUE;
}

// Ignore unsupported scalar attributes. Return SXF_TRUE.
int UAL_SXF_ElemEmpty::setScalarValue(double)
{
  return SXF_TRUE;
}
