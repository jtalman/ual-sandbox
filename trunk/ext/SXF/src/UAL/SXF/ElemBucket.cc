#include "UAL/SXF/ElemBucket.hh"

// Constructor.
UAL::SXFElemBucket::SXFElemBucket(SXF::OStream& out, const char* type, 
				       SXF::ElemBucketHash* hash)
  : SXF::ElemBucket(out, type, hash)
{
}

int UAL::SXFElemBucket::isDeviation() const
{
  return SXF_FALSE;
}

// Do nothing.
int UAL::SXFElemBucket::openObject(const char*)
{
  return SXF_TRUE;
}

// Do nothing. 
void UAL::SXFElemBucket::update()
{
}

// Do nothing.
void UAL::SXFElemBucket::close()
{
}

// Return SXF_FALSE.
int UAL::SXFElemBucket::openArray()
{
  return SXF_FALSE;
}

// Do nothing.
void UAL::SXFElemBucket::closeArray()
{
}

// Return SXF_FALSE.
int UAL::SXFElemBucket::openHash()
{
  return SXF_FALSE;
}

// Do nothing.
void  UAL::SXFElemBucket::closeHash()
{
}

// Return SXF_FALSE.
int UAL::SXFElemBucket::setScalarValue(double)
{
  return SXF_FALSE;
}

// Return SXF_FALSE.
int UAL::SXFElemBucket::setArrayValue(double)
{
  return SXF_FALSE;
}

// Return SXF_FALSE.
int UAL::SXFElemBucket::setHashValue(double, int)
{
  return SXF_FALSE;
}

// Return a number of SMF entry buckets. 
int UAL::SXFElemBucket::getEntrySize() const
{
  return 0;
}

// Return a number of SMF body buckets. 
int UAL::SXFElemBucket::getBodySize() const
{
  return 0;
}

// Return a number of SMF exit buckets. 
int UAL::SXFElemBucket::getExitSize() const
{
  return 0;
}

// Return a SMF entry bucket of element attributes.
PacElemBucket* UAL::SXFElemBucket::getEntryBucket(int index)
{
  return 0;
}

// Return a SMF body bucket of element attributes.
PacElemBucket* UAL::SXFElemBucket::getBodyBucket(int index)
{
  return 0;
}

// Return a SMF exit bucket of element attributes.
PacElemBucket* UAL::SXFElemBucket::getExitBucket(int index)
{
  return 0;
}

// Write data.
void UAL::SXFElemBucket::write(ostream&, const PacLattElement&, const string&)
{
}
