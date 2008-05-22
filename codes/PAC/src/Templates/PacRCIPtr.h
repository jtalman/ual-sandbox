// Library     : PAC
// File        : Templates/PacRCIPtr.h
// Copyright   : see Copyright file
// Description : PacRCIPtr<T> implements a smart pointer that supports 
//               reference counting. The design is based on ideas presented 
//               in Scott Meyers' book, "More Effective C++" (Item 28 - 29).
// Author      : Nikolay Malitsky  


#ifndef PAC_RCI_PTR_H
#define PAC_RCI_PTR_H

#include <iostream>
#include "PAC/Common/PacException.h"

template<class T> class PacRCIPtr 
{
public:

// constructor & destructors

  PacRCIPtr(T* realPtr = 0);
  PacRCIPtr(const PacRCIPtr<T>& rhs); 
  ~PacRCIPtr() {removeReference();}

// copy

  PacRCIPtr& operator=(T* realPtr);
  PacRCIPtr& operator=(const PacRCIPtr<T>& rhs);

// access

  T* operator->() const;             
  T& operator*()  const;

  int count() const;

protected:

  class CountHolder
  {
  public:
    CountHolder()
      : count(0) { pointee = 0;}
    CountHolder(const CountHolder& ch) 
      : count(ch.count) { pointee = ch.pointee; }
    ~CountHolder() { if(pointee)  delete pointee; }

    int count;
    T *pointee;
  };
  
  CountHolder *counter;

private:

  void removeReference(); 
  void createHolder(T* realPtr);                
};

// Constructors 

template<class T>
PacRCIPtr<T>::PacRCIPtr(T* realPtr)
{
  createHolder(realPtr);
}

template<class T>
PacRCIPtr<T>::PacRCIPtr(const PacRCIPtr<T>& rhs)
  : counter(rhs.counter)
{
  if(counter) ++(counter->count);  
}

// Copy operator

template<class T>
PacRCIPtr<T>& PacRCIPtr<T>::operator=(T* realPtr)
{
  removeReference();
  createHolder(realPtr);
  return *this;
}

template<class T>
PacRCIPtr<T>& PacRCIPtr<T>::operator=(const PacRCIPtr<T>& rhs)
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
T* PacRCIPtr<T>::operator->() const
{
 if(!counter) {
   std::string msg = "Error : PacRCIPtr<T>::operator->() : pointer is not allocated \n";
   PacAllocError(msg).raise();
 }
 return counter->pointee;
}

template<class T>
T& PacRCIPtr<T>::operator*() const
{
 if(!counter) {
   std::string msg = "Error : PacRCIPtr<T>::operator*() : pointer is not allocated \n";
   PacAllocError(msg).raise();
 }
 return *(counter->pointee);
}

template<class T>
int PacRCIPtr<T>::count() const
{
  return (counter != 0 ? counter->count : 0);
}

// Protected methods

template<class T>
void PacRCIPtr<T>::createHolder(T* realPtr)
{
  if(realPtr){
    counter = new CountHolder;
    if(!counter) {
      std::string msg = "Error : PacRCIPtr<T>::createHadle(T* realPtr) : allocation failure \n";
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
void PacRCIPtr<T>::removeReference()
{
  if(counter)
    if(--(counter->count) == 0) {
      delete counter;
      counter = 0;
    }
}

#endif



