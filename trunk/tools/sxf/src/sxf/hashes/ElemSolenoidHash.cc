#include "sxf/hashes/ElemSolenoidHash.hh"

// Constructor
SXF::ElemSolenoidHash::ElemSolenoidHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemSolenoidHash::index(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"ks",    KS},
  };

  int index, result = -1;

  if(!str) return result;

  switch (str[0]) {
  case 'k':
    index = 0;
    break;
  default:
    return result;
  }

  char *s = keys[index].name;
  if(!strcmp(str + 1, s + 1)) result = keys[index].number;

  return result;
}
