#include "sxf/hashes/ElemBendHash.hh"

// Constructor
SXF::ElemBendHash::ElemBendHash() 
  : SXF::ElemBucketHash()
{}

// Map an attribute key to its enum value
int SXF::ElemBendHash::index(const char* str) const
{
  static const struct SXF_Key keys[] =
  {
    {"kl",    KL},
    {"kls",   KLS},
    {"fint",  FINT},
    {"hgap",  HGAP},
    {"e1",    E1},
    {"e2",    E2},
  };

  int index, result = -1;

  if(!str) return result;

  if(strlen(str) > 2){
    switch (str[2]) {
    case 's':
      index = KLS;
      break;      
    case 'n':
      index = FINT;
      break;  
    case 'a':
      index = HGAP;
      break; 
    default:
    return result;
    }
  }
  else {
    switch (str[1]) {
    case 'l':
      index = KL;
      break;
    case '1':
      index = E1;
      break;
    case '2':
      index = E2;
      break;
    default:
      return result;
    }
  }

  char *s = keys[index].name;
  if(!strcmp(str, s)) result = keys[index].number;

  return result;
}
