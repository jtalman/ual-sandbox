#include "sxf/hashes/ElemCollimatorHash.hh"

// Constructor
SXF::ElemCollimatorHash::ElemCollimatorHash() 
  : SXF::ElemBucketHash()
{}

// Map an collimator attribute key to its enum value
int SXF::ElemCollimatorHash::index(const char* str) const 
{
  static const struct SXF_Key keys[] =
  {
    {"xsize",     XSIZE},
    {"ysize",     YSIZE},
  };

  int index, result = -1;

  if(!str) return result;

  switch (str[0]) {
  case 'x':
    index = 0;
    break;
  case 'y':
    index = 1;
    break;
  default:
    return result;
  }

  char *s = keys[index].name;
  if(!strcmp(str + 1, s + 1)) result = keys[index].number;

  return result;
}
