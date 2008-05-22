#include "sxf/hashes/ElemAlignHash.hh"

// Constructor.
SXF::ElemAlignHash::ElemAlignHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value.
int SXF::ElemAlignHash::index(const char* str) const 
{
  static const struct SXF_Key keys[] =
  {
    {"al",    AL}
  };

  int index, result = -1;

  if(!str) return result;

  index = AL;
  char *s = keys[index].name;
  if(!strcmp(str + 1, s + 1)) result = keys[index].number;

  return result;
}
