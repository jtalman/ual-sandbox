// Library     : PAC
// File        : SMF/PacGenElement.cc
// Copyright   : see Copyright file
// Description : Implementation of PacGenElement.
// Author      : Nikolay Malitsky

#include "SMF/PacElemKeys.h"
#include "SMF/PacGenElements.h"

PacGenElement::PacGenElement(const string& name, int k)
  : _ptr(name)
{
  if(!name.empty()) {
    _ptr->_key = k;
    if(!PacGenElements::instance()->insert(*this)) {  
     string msg = "Error : PacGenElement(const string& name) : insertion failed for ";
     PacDomainError(msg).raise();
    }
  }
}

PacGenElement* PacGenElement::operator()(const string& n) const
{
  PacGenElements::iterator i = PacGenElements::instance()->find(n);
  if(i == PacGenElements::instance()->end()) return 0;
  return &(*i);
}

const string& PacGenElement::type() const 
{
  int key = this->key();
  PacElemKeys::iterator it = PacElemKeys::instance()->find(key);
  if(it == PacElemKeys::instance()->end()) return  PacElemKey::s_notype;
  return it->name();
}


// Private methods

void PacGenElement::checkName()
{ 
  if(!name().empty()){
    string msg = "Error: PacGenElement::checkName() : attempt to modify PacGenElement ";
    PacDomainError(msg).raise();
  }
}

void PacGenElement::checkPart(int i) const 
{ 
  if(i < 0 || i > 2 ){
      string msg = "Error: PacGenElement::checkPart(i) : i is out of [0, 2] for  ";
      PacDomainError(msg).raise();
  }
}


PacElemPart* PacGenElement::create(int n)
{
  if(_ptr->_parts[n]) return _ptr->_parts[n];

  _ptr->_parts[n] = new PacElemPart();
  if(!_ptr->_parts[n]){
      string msg = "Error: PacGenElement::create(int part) : allocation failure for  ";
      PacDomainError(msg + name()).raise();
  }  
  return  _ptr->_parts[n];
}



