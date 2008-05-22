#include "echo_sxf/EchoElemBucket.hh"

// Constructor.
SXF::EchoElemBucket::EchoElemBucket(SXF::OStream& out, 
				       const char* type, 
				       SXF::ElemBucketHash* hash)
  : SXF::ElemBucket(out, type, hash)
{
}

// Do nothing. 
int SXF::EchoElemBucket::openObject(const char*)
{
  return SXF_TRUE;
}

// Do nothing.
void SXF::EchoElemBucket::update()
{
}

// Do nothing.
void SXF::EchoElemBucket::close()
{
}

// Do nothing.
int SXF::EchoElemBucket::openArray()
{
  return SXF_TRUE;
}

// Do nothing.
void SXF::EchoElemBucket::closeArray()
{
}

// Do nothing.
int SXF::EchoElemBucket::openHash()
{
  return SXF_TRUE;
}

// Do nothing.
void  SXF::EchoElemBucket::closeHash()
{
}

// Do nothing.
int SXF::EchoElemBucket::setScalarValue(double)
{
  return SXF_TRUE;
}

// Do nothing.
int SXF::EchoElemBucket::setArrayValue(double)
{
  return SXF_TRUE;
}

// Do nothing.
int SXF::EchoElemBucket::setHashValue(double, int)
{
  return SXF_TRUE;
}

