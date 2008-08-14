// Library     : PAC
// File        : SMF/SmfElemBucketKey.cc
// Copyright   : see Copyright file
// Description : Implementation of PacElemBucketKey
// Author      : Nikolay Malitsky

// #include <String.h>
#include "SMF/PacElemBucketKeys.h"
#include <string.h>


PacElemBucketKey::PacElemBucketKey(const char*  name, 
   int key, int size, int order, const char* attNames)
  : _ptr(new PacElemBucketKey::Data())
{

  _ptr->_key   = key;
  _ptr->_size  = size;
  _ptr->_order = order;
  _ptr->_name  = name;

  create(attNames);

  if(!PacElemBucketKeys::instance()->insert(*this)) {  
    std::string msg =  "Error : PacElemBucketKey(const char*  name, int key, int size, int order) : ";
           msg += "insertion failed for ";
    PacDomainError(msg + name).raise();
  }

}

PacElemBucketKey* PacElemBucketKey::operator()(int key) const
{
  PacElemBucketKeys::iterator i = PacElemBucketKeys::instance()->find(key);
  if(i == PacElemBucketKeys::instance()->end()) return 0;
  return &(*i);
}

const PacElemAttribKey& PacElemBucketKey::operator[](int index) const
{  
  if(index < 0 || index >= size()) {
    std::string msg =  "Error : PacElemBucketKey::operator[](int index) : ";
           msg += "index is out of [0, size[ for bucket  ";
    PacDomainError(msg + name()).raise();
  }
  return (_ptr->_attKeys)[index];
}

// Private methods

void PacElemBucketKey::checkName()
{
  if(!name().empty()){
    std::string msg = "Error: PacElemBucketKey::checkName() : attempt to modify PacElemBucketKey ";
    PacDomainError(msg + name()).raise();
  }
}

void PacElemBucketKey::create(const char*  names)
{

  /*
  String* _names = new String[size()];
  String tnames(names);

  if(size() > 1) split(tnames, _names, size(), String(" "));
  else          _names[0] = tnames;

  _ptr->_attKeys = new PacElemAttribKey[size()];  

  if(!_ptr->_attKeys) {
    string msg  = "Error : PacElemBucketKey::create(const String& names) : ";
           msg += "allocation failure for bucket ";
    PacDomainError(msg + name()).raise();
  }

  for(int i=0; i < size(); i++){
    _ptr->_attKeys[i].define(*this, _names[i].chars(), i );
  }

  delete [] _names;
  */

  char tnames[120]; strcpy(tnames, names);
  char *key;
  _ptr->_attKeys = new PacElemAttribKey[size()];

  key = strtok(tnames, " ");
  _ptr->_attKeys[0].define(*this, key, 0 );

  for(int i=1; i < size(); i++){
    key = strtok(0, " \0");
   _ptr->_attKeys[i].define(*this, key, i );
  }

}

PacElemBucketKey::Data::~Data()
{
  if(_attKeys) delete [] _attKeys;
}

// PacKeyOfElemBucketKey

void PacKeyOfElemBucketKey::operator()(PacElemBucketKey&, int) const
{
  std::string msg  = "Error : PacKeyOfElemBucketKey::operator(PacElemBucketKey& x, int key) const :";
         msg += "don't insert items in collection \n";
  PacDomainError(msg).raise();
}

int PacKeyOfElemBucketKey::count(const PacElemBucketKey&) const
{
  std::string msg  = "Error : PacKeyOfElemBucketKey::count(const PacElemBucketKey& x ) const :";
         msg += "don't erase items from collection \n";
  PacDomainError(msg).raise();
  return 0;
}

// PacElemAttribKey

PacElemAttribKey PacElemAttribKey::operator()(int order) const 
{
  if(!_bucketKey.order()){
    std::string msg = "Error : PacElemAttribKey::operator()(int order) : attempt to define order for  ";    
    PacDomainError(msg + _name).raise();
  }    
  PacElemAttribKey tmp = *this;
  tmp._order = order;
  return tmp;
}

void PacElemAttribKey::checkName()
{
  if(!name().empty()){
    std::string msg = "Error: PacElemAttribKey::checkName() : attempt to modify PacElemAttribKey ";
    PacDomainError(msg + name()).raise();
  }
}

void PacElemAttribKey::define(const PacElemAttribKey& key)
{
  checkName();

  _bucketKey = key._bucketKey;

  _name  = key._name;
  _index = key._index;
  _order = key._order;

}

void PacElemAttribKey::define(const PacElemBucketKey& bucketKey, const std::string& name, int index)
{
  checkName();

  _bucketKey = bucketKey;

  _name  = name;
  _index = index;

  _order =  0;

}

PacElemBucketKey pacLength("Length", PAC_LENGTH, 1, 0, "L");
PacElemBucketKey pacBend("Bend", PAC_BEND, 2, 0, "ANGLE FINT");
PacElemBucketKey pacMultipole("Multipole", PAC_MULTIPOLE, 2, 1, "KL KTL");
PacElemBucketKey pacOffset("Offset", PAC_OFFSET, 3, 0, "DX DY DS");
PacElemBucketKey pacRotation("Rotation", PAC_ROTATION, 3, 0, "DPHI DTHETA TILT");
PacElemBucketKey pacAperture("Aperture", PAC_APERTURE, 3, 0, "SHAPE XSIZE YSIZE");
PacElemBucketKey pacComplexity("Complexity", PAC_COMPLEXITY, 1, 0, "N");
PacElemBucketKey pacSolenoid("Solenoid", PAC_SOLENOID, 1, 0, "KS");
PacElemBucketKey pacRfCavity("RfCavity", PAC_RFCAVITY, 3, 1, "VOLT LAG HARMON");





