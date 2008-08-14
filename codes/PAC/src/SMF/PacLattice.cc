// Library     : PAC
// File        : SMF/SmfLattice.cc
// Copyright   : see Copyright file
// Description : The implementation of the class PacLattice.
// Author      : Nikolay Malitsky

// #include <String.h>
#include "SMF/PacLattices.h"
#include "pcre.h"

#include <string.h>

#define OVECCOUNT 30    /* should be a multiple of 3 for PCRE*/

// Constructors & copy operator

PacLattice::PacLattice()
  : _ptr(new PacLattice::Data())
{
}

PacLattice::PacLattice(const string& name)
  : _ptr(name)
{
  if(name.empty()) {
     string msg = "Error : PacLattice(const string& name) : name is empty  ";
     PacDomainError(msg + name).raise();
  }

  if(!PacLattices::instance()->insert(*this)) {  
     string msg = "Error : PacLattice(const string& name) : insertion failed for ";
     PacDomainError(msg + name).raise();
  }

}

PacLattice::PacLattice(const PacLattice& la)
  : _ptr(la._ptr)
{
}

void PacLattice::operator = (const PacLattice& la) 
{ 
  checkName(); 
  _ptr = la._ptr;
}

// Modifiers

void PacLattice::set(PacList<PacLattElement>& array)
{
  check(); 

  if(_ptr->_vector) delete [] _ptr->_vector;
  _ptr->_vector = 0;

  _ptr->_size = array.size();

  if(_ptr->_size){
    _ptr->_vector = new PacLattElement[_ptr->_size];

    int i = 0;
    for(PacList<PacLattElement>::iterator it = array.begin(); it != array.end(); it++){
      _ptr->_vector[i++] = *it;
    }
  }
  
}

void PacLattice::set(const PacLattice& la)
{
  check(); 
  setLattice(la, PacLattice());
} 

void PacLattice::set(PacLine& li)
{
  check(); 
  setLattice(li);
}

void PacLattice::add(const PacLattice& la)
{
  check();

  PacLattice sum;
  sum.setLattice(*this, la);

  // Line

  // Vector

  setVector(sum, PacLattice());

  // Lattices

  addLattice(la);

}

void PacLattice::erase()
{
  check(); 
  eraseLattice(); 
}


PacLattice  operator,(const PacLattice& l1, const PacLattice& l2)
{
  PacLattice lattice;
  lattice.setLattice(l1, l2);
  return lattice;
}

// Access methods

// ... to elements

PacVector<int> PacLattice::indexes(const char* name)
{
  pcre *regex=NULL;  //pointer to regular expression for pattern matching
  pcre_extra *study=NULL; //pointer to regular expression for pattern matching
  int match;  //result from pattern matching
  const char *error;
  int erroffset;
  int ovector[OVECCOUNT];
  const char *subject;
  int length;

  PacVector<int> ind(0);

  //parse the regex
  regex = pcre_compile(name, 0, &error, &erroffset, NULL); 
  if (regex == NULL){
    std::cerr<<"Error:  PacLattice::indexes(const char* name) - PCRE compilation failed at offset "<<erroffset<<": "<<error<<std::endl;
    return ind;
  }
 
  study=pcre_study(regex, 0, &error); 
  if (study == NULL && error!=NULL){
    std::cerr<<"Error:  PacLattice::indexes(const char* name) -  PCRE study failed wwith error: "<<error<<std::endl;
    //   delete regex;
    pcre_free(regex);
    return ind;
  }
 
  //loop over lattice and find all matches,
  for(int i=0; i<size(); i++){
    subject=(*this)[i].getDesignName().c_str();
    length=strlen(subject);

    match = pcre_exec(regex, study, subject, length, 0, 0, ovector, OVECCOUNT);
    if(match>0) ind.push_back(i); //match the regex  
   }


  //  delete regex;
  //  if(study != NULL) delete study;
  pcre_free(regex);
  if(study != NULL) pcre_free(study);

  return ind;
}

// ... to collection items

PacLattice* PacLattice::operator()(const string& key) const
{
  PacLattices::iterator i = PacLattices::instance()->find(key);
  if(i == PacLattices::instance()->end()) return 0;
  return &(*i);
}

// Private methods

// Lattice

void PacLattice::setLattice(const PacLattice& l1, const PacLattice& l2)
{
  eraseLattice();

  // Line

  // Vector

  setVector(l1, l2);

  // Lattices

  addLattice(l1);
  addLattice(l2);

}


void PacLattice::setLattice(PacLine& li)
{
  eraseLattice();

  // Line

  check(li);
  _ptr->_line = li.name();
  li.lock();

  // Vector 

  setVector(li);

}

void PacLattice::addLattice(const PacLattice& la)
{
  if(!line().empty()) {
    string msg = "Error : PacLattice::addLattice(const PacLattice& la) : ";
           msg += "attempt to modify lattice that is defined by line \n";
    PacDomainError(msg + line()).raise();
  }

  if(!la.name().empty()) _ptr->_lattices.push_back(la.name());
  else{
    for(PacLattice::const_iterator it = la.lattices().begin(); it != la.lattices().end(); it++)
      _ptr->_lattices.push_back(*it);
  }

}

void PacLattice::eraseLattice()
{

  // Line
  
  _ptr->_line = "";

  // Vector

  if(_ptr->_vector) delete [] _ptr->_vector;
  _ptr->_vector = 0;
  _ptr->_size = 0;

  // Lattices

  _ptr->_lattices.erase(_ptr->_lattices.begin(), _ptr->_lattices.end());

}

void PacLattice::check()
{
   if(_ptr.count() == 0 ){
    string msg = "Error: PacLattice::check() : the line doesn't have a name ";
    PacDomainError(msg).raise();
  }

  if(isLocked()){
    string msg = "Error: PacLattice::check() : it is the locked lattice  ";
    PacDomainError(msg + name()).raise();
  }
}

void PacLattice::checkName()
{ 
  if(!name().empty()){
    string msg =  "Error: PacLattice::checkName() : ";
           msg += "attempt to change a name for the lattice ";
    PacDomainError(msg + name()).raise();
  }
}

// Vector


void PacLattice::setVector(const PacLattice& l1, const PacLattice& l2)
{
  if(_ptr->_vector) delete [] _ptr->_vector;
  _ptr->_vector = 0;

  _ptr->_size = l1.size() + l2.size();

  if(_ptr->_size){
    _ptr->_vector = new PacLattElement[_ptr->_size];

    if(!_ptr->_vector) {
      string msg  = "Error: PacLattice::setVector(const PacLattice& l1, const PacLattice& l2) :";
             msg += "allocation failure \n";;
      PacAllocError(msg).raise();
    }
    int i;
    for(i=0; i < l1.size(); i++) _ptr->_vector[i]             = l1._ptr->_vector[i];
    for(i=0; i < l2.size(); i++) _ptr->_vector[l1.size() + i] = l2._ptr->_vector[i];
  }
}

void PacLattice::setVector(const PacLine& li)
{

  if(_ptr->_vector) delete [] _ptr->_vector;
  _ptr->_vector = 0;

  _ptr->_size = count(li);

  if(_ptr->_size){
    _ptr->_vector = new PacLattElement[_ptr->_size];

    if(!(_ptr->_vector)) {
      string msg  = "Error: PacLattice::setVector(const PacLine& li) :";
             msg += " allocation failure \n";
      PacAllocError(msg).raise();
    }

    int counter = 0;
    track(li, 1, counter);
  }
}

// Line

void PacLattice::check(const PacLine& li)
{
  if(li.name().empty()) {
    string msg  = "Error: PacLattice::check(const PacLine& li) : ";
           msg += "li doesn't have name \n";
    PacDomainError(msg).raise();
  }
}

int PacLattice::count(const PacLine& li)
{
  int s = 0;
  for(PacLine::const_iterator it = li.list().begin(); it != li.list().end(); it++){
     if((*it).element().count()) s += abs((*it).repetition());
     else                        s += abs((*it).repetition())*count((*it).line());
  }

  return s;
}

void PacLattice::track(const PacLine& li, int direction, int& counter)
{
  int p;
  if(direction > 0){
    for(PacLine::const_iterator it = li.list().begin(); it != li.list().end(); it++)
      for(p = 0; p < abs((*it).repetition()); p++){
	if((*it).element().count()) _ptr->_vector[counter++] = (*it).element();
	else                        track((*it).line(), direction*(*it).repetition(), counter);
      }
  }
  else{
    for(PacLine::const_reverse_iterator it = li.list().rbegin(); it != li.list().rend(); it++)
      for(p = 0; p < abs((*it).repetition()); p++){
	if((*it).element().count()) _ptr->_vector[counter++] = (*it).element();
	else                        track((*it).line(), direction*(*it).repetition(), counter);
      }  
  }  
}
