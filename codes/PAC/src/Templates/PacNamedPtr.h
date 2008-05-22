// Library     : PAC
// File        : Templates/PacNamedPtr.h
// Copyright   : see Copyright file
// Description : PacNamedPtr<T> implements a named smart pointer 
//               that supports reference counting. The design is 
//               based on ideas presented in Scott Meyers' book, 
//               "More Effective C++" (Item 28 - 29).
// Author      : Nikolay Malitsky  

#ifndef PAC_NAMED_PTR_H
#define PAC_NAMED_PTR_H

#include <iostream>
#include <string>

#include "PAC/Common/PacException.h"

template<class T> class PacNamedPtr 
{
public:

// constructor & destructors

  PacNamedPtr(T* realPtr = 0);
  PacNamedPtr(const std::string& n);
  PacNamedPtr(const PacNamedPtr<T>& rhs); 
  ~PacNamedPtr() {removeReference();}

// copy

  PacNamedPtr& operator=(const PacNamedPtr<T>& rhs);

// access

  T* operator->() const;             
  T& operator*()  const;

  const std::string& name() const;
  int count() const;

protected:

  class CountHolder
  {
  public:
    CountHolder()
      : count(0), pointee(0) {}
    CountHolder(const CountHolder& ch) 
      : count(ch.count), name(ch.name), pointee(ch.pointee) {}
    ~CountHolder() { if(pointee)  delete pointee; }

    int count;
    std::string name;
    T *pointee;
  };  

  CountHolder *counter;

private:

  static std::string& empty_name();
  void removeReference();        
};

template<class T> 
std::string& PacNamedPtr<T>::empty_name() { 
  static std::string _empty_name;
  return _empty_name;
}


// Constructors 

template<class T>
PacNamedPtr<T>::PacNamedPtr(T* realPtr)
{
  if(realPtr){
    counter = new CountHolder;
    if(!counter) {
      std::string msg = "Error : PacNamedPtr<T>::PacNamedPtr(T* realPtr) : allocation failure \n";
      PacAllocError(msg).raise();
    }
    counter->pointee = realPtr;
    ++(counter->count);
  }
  else {
    counter = 0;
  }
}

template<class T>
PacNamedPtr<T>::PacNamedPtr(const std::string& n)
{
  if(!n.empty()){
    counter = new CountHolder;
    if(!counter) {
      std::string msg = "Error : PacNamedPtr<T>::PacNamedPtr(const string& n) : allocation failure for ";
      PacAllocError(msg + n).raise();
    }
    counter->pointee = new T;
    if(!counter->pointee){
      std::string msg = "Error : PacNamedPtr<T>::PacNamedPtr(const string& n) : allocation failure for ";
      PacAllocError(msg + n).raise();
    }
    counter->name = n;
    ++(counter->count);
  }
  else {
    counter = 0;
  }
}

template<class T>
PacNamedPtr<T>::PacNamedPtr(const PacNamedPtr<T>& rhs)
  : counter(rhs.counter)
{
  if(counter) ++(counter->count);  
}

// Copy operator

template<class T>
PacNamedPtr<T>& PacNamedPtr<T>::operator=(const PacNamedPtr<T>& rhs)
{
  if(counter != rhs.counter) {    
    removeReference();                
    counter = rhs.counter;
    if(counter) ++(counter->count);
  }
  
  return *this;
}

// Access

template<class T> 
T* PacNamedPtr<T>::operator->() const
{
 if(!counter) {
   std::string msg = "Error : PacNamedPtr<T>::operator->() : pointer is not allocated \n";
   PacAllocError(msg).raise();
 }
 return counter->pointee;
}

template<class T>
T& PacNamedPtr<T>::operator*() const
{
 if(!counter) {
   std::string msg = "Error : PacNamedPtr<T>::operator->() : pointer is not allocated \n";
   PacAllocError(msg).raise();
 }
 return *(counter->pointee);
}

template<class T>
const std::string& PacNamedPtr<T>::name() const
{
  return (counter != 0 ? counter->name : empty_name());
}

template<class T>
int PacNamedPtr<T>::count() const
{
  return (counter != 0 ? counter->count : 0);
}

template<class T>
void PacNamedPtr<T>::removeReference()
{
  if(counter)
    if(--(counter->count) == 0) {
      delete counter;
      counter = 0;
    }
}

#endif


