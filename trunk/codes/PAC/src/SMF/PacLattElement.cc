// Library     : PAC
// File        : SMF/PacLattElement.cc
// Copyright   : see Copyright file
// Description : Implementation of PacLattElement.
// Author      : Nikolay Malitsky

#include "SMF/PacElemKeys.h"
#include "SMF/PacElemComplexity.h"
#include "SMF/PacElemLength.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacLattElement.h"


// New interface

void PacLattElement::setPosition(double at)
{
  _ptr->_at = at;
}

double PacLattElement::getPosition() const
{
  return _ptr->_at;
}

// Private methods

void PacLattElement::check(const PacGenElement& e)
{
  if(e.name().empty()){
   string msg = "Error: PacLattElement::check(const PacGenElement& e) : e doesn't have name \n";
   PacDomainError(msg).raise();
  }
}

void PacLattElement::define(const PacGenElement& e)
{
  check(e);
  _ptr->_genElement = e;

  _ptr->_key = e.key();
  _ptr->_map = e.map();

  for(int i=0; i < 3; i++) 
    if(e.getPart(i)) *(setPart(i)) = e.getPart(i)->attributes();

}

void PacLattElement::checkGenElement()
{ 
  if(genElement().count()){
    string msg  = "Error: PacLattElement::checkGenElement() : ";
           msg += "attempt to change the genElement for lattElement ";
    PacDomainError(msg + name()).raise();
  }
}



void PacLattElement::checkPart(int i) const 
{ 
  if(i < 0 || i > 2 ){
      string msg = "Error: PacLattElement::checkPart(i) : i is out of [0, 2] for  ";
      PacDomainError(msg + name()).raise();
  }
}


PacElemAttributes* PacLattElement::create(int n)
{
  if(_ptr->_parts[n]) return _ptr->_parts[n];

  _ptr->_parts[n] = new PacElemAttributes();
  if(!_ptr->_parts[n]){
      string msg = "Error: PacLattElement::create(int part) : allocation failure for  ";
      PacDomainError(msg + name()).raise();
  }  
  return  _ptr->_parts[n];
}

double PacLattElement::getLength() const
{
  PacElemAttributes* body = getBody();

  if(!body) { return 0.0; }

  return body->get(PAC_L);
}

void PacLattElement::addLength(double l)
{
  PacElemAttributes* body = getBody();

  if(!body) return;

  PacElemLength elemLength;
  elemLength.l(l);

  body->add(elemLength);
}

double PacLattElement::getAngle() const
{
  PacElemAttributes* body = getBody();

  if(!body) { return 0.0; }

  return body->get(PAC_ANGLE);
}

void PacLattElement::addAngle(double angle)
{
  PacElemAttributes* body = getBody();

  if(!body) return;

  PacElemBend elemBend;
  elemBend.angle() = angle;

  body->add(elemBend);
}

double PacLattElement::getN() const
{
  PacElemAttributes* body = getBody();

  if(!body) { return 0.0; }

  return body->get(PAC_N);
}

void PacLattElement::addN(double n)
{
  PacElemAttributes* body = getBody();

  if(body == 0) { create(1); }

  PacElemComplexity complexity;
  complexity.n() = n;
  body->add(complexity);
}

const string& PacLattElement::type() const 
{
  int key = this->key();
  PacElemKeys::iterator it = PacElemKeys::instance()->find(key);
  if(it == PacElemKeys::instance()->end()) return PacElemKey::s_notype;
  return it->name();
}

