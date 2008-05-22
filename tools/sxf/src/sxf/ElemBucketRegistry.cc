#include "sxf/ElemBucketRegistry.hh"

// #include <iostream.h>

// Constructor.
SXF::ElemBucketRegistry::ElemBucketRegistry(SXF::OStream& out) 
  : m_refOStream(out),
    m_iSize(0), 
    m_aBuckets(0),    
    m_pErrorBucket(0)
{
}

// Destructor.
SXF::ElemBucketRegistry::~ElemBucketRegistry() {
}

// Register a particular bucket reader.
SXF::ElemBucket* SXF::ElemBucketRegistry::bind(const char* type, 
					     SXF::ElemBucket* bucket)
{
  int index  = hash(type);

  if(!bucket) {
    m_refOStream.cfe_error() 
      << "\n *** CFE Error: SXF::ElemBucketRegistry::bind : bucket (" 
      << type << ") is NULL" << endl;
    return 0;
  }

  if(index < 0) {
    m_refOStream.cfe_error() 
      << "\n *** CFE Error: SXF::ElemBucketRegistry::bind : bucket (" 
      << type << ") is not recognized" << endl;
    return 0;
  }

  if(m_aBuckets[index]){
    m_refOStream.cfe_error() 
      << "\n *** CFE Error: SXF::ElemBucketRegistry::bind : bucket (" 
      << type << ") has been defined" << endl;
    return 0;
  }    

  return m_aBuckets[index] = bucket;
}

// Select a bucket reader.
SXF::ElemBucket* SXF::ElemBucketRegistry::getBucket(const char* type)
{
  int index  = hash(type); 

  SXF::ElemBucket* result = 0;
  if(index >= 0) { result = m_aBuckets[index];}

  return result;
}

SXF::ElemBucket* SXF::ElemBucketRegistry::getErrorBucket()
{
  return m_pErrorBucket;
}

// Map a bucket type to its index in the array of bucket readers.
int SXF::ElemBucketRegistry::hash(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"entry",    0},
    {"exit",     1},
    {"align",    2},
    {"aperture", 3}
  };

  int index, result = -1;

  if(!str) return result;

  switch (str[1]) {
  case 'n':
    index = 0;
    break;
  case 'x':
    index = 1;
    break;
  case 'l':
    index = 2;
    break;
  case 'p':
    index = 3;
    break;
  default:
    return result;
  }
  
  char *s = keys[index].name;
  if(!strncmp(str, s, strlen(s))) result = keys[index].number;

  return result;
}

// Allocate an array of element bucket readers.
void SXF::ElemBucketRegistry::allocateRegistry()
{
  m_iSize = 4;
  m_aBuckets = new SXF::ElemBucket*[m_iSize];
  for(int i=0; i < m_iSize; i++){ m_aBuckets[i] = 0; }
}
