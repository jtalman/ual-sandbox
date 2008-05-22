#include "sxf/hashes/ElemMultipoleHash.hh"

// Constructor
SXF::ElemMultipoleHash::ElemMultipoleHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemMultipoleHash::index(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"kl",    KL},
    {"kls",   KLS},
    {"lrad",  LRAD},
  };

  int index, result = -1;

  if(!str) return result;

  switch (strlen(str)) {
  case 2: 
    index = 0;
    break;
  case 3:
    index = 1;
    break;
  case 4:
    index = 2;
    break;
  default:
    return result;   
  };

  char *s = keys[index].name;
  if(!strcmp(str, s)) result = keys[index].number;

  return result;
}
