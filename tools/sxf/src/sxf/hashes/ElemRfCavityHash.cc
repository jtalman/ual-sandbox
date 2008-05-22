#include "sxf/hashes/ElemRfCavityHash.hh"

// Constructor
SXF::ElemRfCavityHash::ElemRfCavityHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemRfCavityHash::index(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"volt",   VOLT},
    {"lag",    LAG},
    {"harmon", HARMON},
    {"shunt",  SHUNT},
    {"tfill",  TFILL},
  };

  int index, result = -1;

  if(!str) return result;

  switch (str[0]) {
  case 'v':
    index = 0;
    break;
  case 'l':
    index = 1;
    break;
  case 'h':
    index = 2;
    break;
  case 's':
    index = 3;
    break;
  case 't':
    index = 4;
    break;    
  default:
    return result;
  }

  char *s = keys[index].name;
  if(!strcmp(str + 1, s + 1)) result = keys[index].number;

  return result;
}
