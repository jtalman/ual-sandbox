#include "ual_sxf/ElemBucket.hh"

// Constructor.
UAL_SXF_ElemBucket::UAL_SXF_ElemBucket(SXF::OStream& out, const char* type, 
				       SXF::ElemBucketHash* hash)
  : SXF::ElemBucket(out, type, hash)
{
}

// Do nothing.
int UAL_SXF_ElemBucket::openObject(const char*)
{
  return SXF_TRUE;
}

// Do nothing. 
void UAL_SXF_ElemBucket::update()
{
}

// Do nothing.
void UAL_SXF_ElemBucket::close()
{
}

// Return SXF_FALSE.
int UAL_SXF_ElemBucket::openArray()
{
  return SXF_FALSE;
}

// Do nothing.
void UAL_SXF_ElemBucket::closeArray()
{
}

// Return SXF_FALSE.
int UAL_SXF_ElemBucket::openHash()
{
  return SXF_FALSE;
}

// Do nothing.
void  UAL_SXF_ElemBucket::closeHash()
{
}

// Return SXF_FALSE.
int UAL_SXF_ElemBucket::setScalarValue(double)
{
  return SXF_FALSE;
}

// Return SXF_FALSE.
int UAL_SXF_ElemBucket::setArrayValue(double)
{
  return SXF_FALSE;
}

// Return SXF_FALSE.
int UAL_SXF_ElemBucket::setHashValue(double, int)
{
  return SXF_FALSE;
}

// Return a number of SMF entry buckets. 
int UAL_SXF_ElemBucket::getEntrySize() const
{
  return 0;
}

// Return a number of SMF body buckets. 
int UAL_SXF_ElemBucket::getBodySize() const
{
  return 0;
}

// Return a number of SMF exit buckets. 
int UAL_SXF_ElemBucket::getExitSize() const
{
  return 0;
}

// Return a SMF entry bucket of element attributes.
PacElemBucket* UAL_SXF_ElemBucket::getEntryBucket(int index)
{
  return 0;
}

// Return a SMF body bucket of element attributes.
PacElemBucket* UAL_SXF_ElemBucket::getBodyBucket(int index)
{
  return 0;
}

// Return a SMF exit bucket of element attributes.
PacElemBucket* UAL_SXF_ElemBucket::getExitBucket(int index)
{
  return 0;
}

// Write data.
void UAL_SXF_ElemBucket::write(ostream&, const PacLattElement&, const string&)
{
}
