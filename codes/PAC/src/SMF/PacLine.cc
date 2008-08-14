// Library     : PAC
// File        : SMF/PacLine.cc
// Copyright   : see Copyright file
// Description : The implementation of the class PacLine.
// Author      : Nikolay Malitsky

#include "SMF/PacLines.h"
#include <string.h>


static PacLine  __pacLine;
PacList<PacLineNode> PacLine::_empty_list;

// PacLineNode

PacLineNode::PacLineNode() 
  : _repetition(1)
{
  create();
}

PacLineNode::PacLineNode(const PacLine& l, int p)
  : _repetition(p)
{ 
  create();
  *_line = l;
}

PacLineNode::PacLineNode(const PacGenElement& e, int p)
  : _repetition(p), _element(e)
{
  check(e);
  create();
}

PacLineNode::PacLineNode(const PacLineNode& n)
{
  create();
  define(n);
}

PacLineNode::~PacLineNode()
{
  if(_line) delete _line;
}


void PacLineNode::operator=(const PacLineNode& n) 
{
  define(n);
}

const PacLine& PacLineNode::line() const
{
  return *_line;
}


void PacLineNode::define(const PacLineNode& n)
{
  _repetition = n._repetition;
  _element = n._element;
  *_line = *n._line;;
}

void PacLineNode::create()
{
  _line = new PacLine();
  if(!_line) PacAllocError("Error: PacLineNode::create() : allocation failure \n").raise();
}

void PacLineNode::check(const PacGenElement& e)
{
  if(e.name().empty()) 
    PacDomainError("Error: PacLineNode::check(const PacGenElement& e) : e doesn't have name \n").raise();
}

// PacLine

PacLine::PacLine(const string& name)
  : _ptr(name)
{
  if(!name.empty()) {
    if(!PacLines::instance()->insert(*this)) {  
      string msg = "Error : PacLine(const string& name) : insertion failed for ";
      PacDomainError(msg + name).raise();
    }
  }
}

PacLine* PacLine::operator()(const string& key) const
{
  PacLines::iterator i = PacLines::instance()->find(key);
  if(i == PacLines::instance()->end()) return 0;
  return &(*i);
}

// Interface

PacLine& operator , (const PacLine& l1, const PacLine& l2)
{

  __pacLine.checkTmpLine(l1, l2);

  if(&l1 == &__pacLine){
    __pacLine._ptr->_list.push_back(PacLineNode(l2));
    return __pacLine;
  }

  if(&l2 == &__pacLine){
    __pacLine._ptr->_list.push_front(PacLineNode(l1));
    return __pacLine;
  }

  __pacLine.create();
  __pacLine._ptr->_list.push_back(PacLineNode(l1));
  __pacLine._ptr->_list.push_back(PacLineNode(l2));

  return __pacLine;
}

PacLine& operator , (const PacGenElement& e1, const PacGenElement& e2)
{
  __pacLine.create();
  __pacLine._ptr->_list.push_back(PacLineNode(e1));
  __pacLine._ptr->_list.push_back(PacLineNode(e2));

  return __pacLine;
}


PacLine& operator , (const PacGenElement& e, const PacLine& l)
{
  if(&l != &__pacLine){
    __pacLine.create();
    __pacLine._ptr->_list.push_back(PacLineNode(l));
  }

  __pacLine._ptr->_list.push_front(PacLineNode(e));

  return __pacLine;
}

PacLine& operator , (const PacLine& l, const PacGenElement& e)
{ 
  if(&l != &__pacLine) {
    __pacLine.create();
    __pacLine._ptr->_list.push_front(PacLineNode(l));
  }

  __pacLine._ptr->_list.push_back(PacLineNode(e));

  return __pacLine;
}


PacLine operator*(int p, PacGenElement& e)
{
  PacLine line;
  line.create();
  line._ptr->_list.push_back(PacLineNode(e, p));
  return line;  
}

PacLine operator*(int p, PacLine& l)
{    
  PacLine line;
  line.checkTmpLine(l, l);
  line.create();
  line._ptr->_list.push_back(PacLineNode(l, p));
  return line; 
}

// Private methods

void PacLine::check()
{
  if(_ptr.count() == 0 ){
    string msg = "Error: PacLine::check() : the line doesn't have a name ";
    PacDomainError(msg).raise();
  }

  if(isLocked()){
    string msg = "Error: PacLine::check() : it is the locked line  ";
    PacDomainError(msg + name()).raise();
  }
}

void PacLine::create()
{
  PacNamedPtr<PacLine::Data> tmp(new PacLine::Data());
  _ptr = tmp;

}

void PacLine::erase()
{
  check();
  if(_ptr.count()) _ptr->_list.erase(_ptr->_list.begin(), _ptr->_list.end());
}


// Private methods

void PacLine::checkName()
{ 
  if(!name().empty()){
    string msg = "Error: PacLine::checkName() : attempt to change a name for the line ";
    PacDomainError(msg + name()).raise();
  }
}


void PacLine::checkTmpLine(const PacLine& l1, const PacLine& l2)
{
  if(&__pacLine == &l1 && &__pacLine == &l2){       
    string msg = "Error: PacLine::checkTmpLine(const PacLine& l1, const PacLine& l2) : &__pacLine == &l1 == &l2 ";
    msg += "don't use parentheses \n";
    PacDomainError(msg).raise();
  }
}

void PacLine::push_back(const PacLine& l)
{
  check(); 
  _ptr->_list.push_back(PacLineNode(l));
}

void PacLine::push_back(const PacGenElement& e)
{
 check();
 _ptr->_list.push_back(PacLineNode(e));  
}
