#include "sxf/hashes/ElemEmptyHash.hh"

// Constructor
SXF::ElemEmptyHash::ElemEmptyHash() 
  : SXF::ElemBucketHash()
{}

// Return -1 for any attributes
int SXF::ElemEmptyHash::index(const char*) const 
{
  return -1;
}
