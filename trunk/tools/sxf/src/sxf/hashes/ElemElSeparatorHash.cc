#include "sxf/hashes/ElemElSeparatorHash.hh"

// Constructor
SXF::ElemElSeparatorHash::ElemElSeparatorHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemElSeparatorHash::index(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"el",    EL},
    {"ex",    EX},
    {"ey",    EY},    
  };

  int index, result = -1;

  if(!str) return result;

  switch (str[1]) {
  case 'l':
    index = 0;
    break;
  case 'x':
    index = 1;
    break;
  case 'y':
    index = 2;
    break;
  default:
    return result;
  }

  char *s = keys[index].name;
  if(!strcmp(str, s)) result = keys[index].number;

  return result;
}
