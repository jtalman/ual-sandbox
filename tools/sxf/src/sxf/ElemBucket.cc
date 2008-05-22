#include "sxf/ElemBucket.hh"

// Constructor.
SXF::ElemBucket::ElemBucket(SXF::OStream& out, const char* type, SXF::ElemBucketHash* hash) 
  : m_refOStream(out), m_iAttribID(-1) 
{
  setType(type);
  m_pHash = hash;
}

// Destructor.
SXF::ElemBucket::~ElemBucket() 
{
  if(m_sType) { delete [] m_sType; }
  if(m_pHash) { delete m_pHash; }
}

// Select a bucket attribute and make it current. 
int SXF::ElemBucket::openAttribute(const char* name)
{
  m_iAttribID = m_pHash->index(name);
  return (m_iAttribID >= 0);
}

// Close a current attribute.
void SXF::ElemBucket::closeAttribute()
{
  m_iAttribID = -1;
}

// Set bucket type.
void SXF::ElemBucket::setType(const char* type)
{
  m_sType = new char[strlen((char*) type) + 1];
  strcpy(m_sType, (char*)  type);
}

