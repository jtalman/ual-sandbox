#include "sxf/hashes/ElemBeamBeamHash.hh"

// Constructor
SXF::ElemBeamBeamHash::ElemBeamBeamHash() 
  : ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemBeamBeamHash::index(const char* str) const 
{
  static const struct SXF_Key keys[] =
  {
    {"sigx",   SIGX},
    {"sigy",   SIGY},
    {"xma",    XMA},
    {"yma",    YMA},
    {"npart",  NPART},
    {"charge", CHARGE},
  };

  int index, result = -1;

  if(!str) return result;

  int length = strlen(str);

  if(length < 4){
    switch (str[0]){
    case 'x':
      index = 2;
      break;
    case 'y':
      index = 3;
      break;
    default:
      return result;
    }
  }
  else if (length == 4){
    switch (str[3]) {
    case 'x':
      index = 0;
      break;
    case 'y':
      index = 1;
      break;
    default:
      return result;
    }
  }
  else {
    switch (str[4]) {
    case 't':
      index = 4;
      break;
    case 'g':
      index = 5;
      break;
    default:
      return result;
    }
  }      

  char *s = keys[index].name;
  if(!strcmp(str, s)) result = keys[index].number;

  return result;
}
