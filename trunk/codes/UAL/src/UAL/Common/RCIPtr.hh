// Library     : UAL
// File        : UAL/Common/RCIPtr.hh
// Copyright   : see Copyright file
// Authors     : implemented after Scott Meyers' approach. See the "More Effective C++" book

#ifndef UAL_RCI_PTR_HH
#define UAL_RCI_PTR_HH

#include <iostream>
#include "UAL/Common/RCObject.hh"

namespace UAL {

   template<class T> class CountHolder : public RCObject {
   public: 
     CountHolder() { m_pointee = 0; }
     ~CountHolder() { delete m_pointee; }
     T* m_pointee;
   };

  /** A smart pointer of RCObject's.*/

  template<class T> class RCIPtr {

  public :

    /** Constructor */
    RCIPtr();

    /** Creates a smart pointer from a real one */    
    RCIPtr(T* realPtr);

    /** Copy constructor */
    RCIPtr(const RCIPtr& rhs);

    /** Destructor */
    ~RCIPtr();

    /** Copy operator */
    RCIPtr& operator=(const RCIPtr& rhs);

    /** Dereferences a smart pointer to return a real pointer */    
    T* operator->() const;

    /** Dereferences a smart pointer */
    T& operator*() const;   

    /** Checks if the real pointer is defined. */    
    bool isValid() const;

  private:

    /*
    struct CountHolder : public RCObject {
      ~CountHolder() { delete m_pointee; }
      T* m_pointee;
    };
    */

    CountHolder<T>* m_counter;
    
    void init();

  };

}

template<class T> 
void UAL::RCIPtr<T>::init()
{
  if(m_counter == 0) {
    return;
  }

  if(!m_counter->isShareable()) {
    // m_counter = new CountHolder();
    // m_counter->m_pointee = m_counter->m_pointee->clone();
    std::cerr << "RCIPtr is not shareable and needs to be cloned " << std::endl;
  }

  m_counter->addReference();
}


template<class T>
UAL::RCIPtr<T>::RCIPtr()
  : m_counter(new UAL::CountHolder<T>())
{
  m_counter->m_pointee = 0;
  init();
}

template<class T>
UAL::RCIPtr<T>::RCIPtr(T* realPtr)
  : m_counter(new UAL::CountHolder<T>())
{
  m_counter->m_pointee = realPtr;
  init();
}

template<class T>
UAL::RCIPtr<T>::RCIPtr(const UAL::RCIPtr<T>& rhs)
  : m_counter(rhs.m_counter)
{
  init();
}

template<class T>
UAL::RCIPtr<T>::~RCIPtr()
{
  if(m_counter != 0 && m_counter->removeReference() <= 0) delete m_counter;
}

template<class T>
UAL::RCIPtr<T>& UAL::RCIPtr<T>::operator=(const UAL::RCIPtr<T>& rhs)
{
  if(m_counter != rhs.m_counter) {
    if(m_counter != 0 && m_counter->removeReference() <= 0) delete m_counter;
    m_counter = rhs.m_counter;
    init();
  }
  return *this;
}

template<class T>
T* UAL::RCIPtr<T>::operator->() const 
{
  return m_counter->m_pointee; 
}

template<class T>
T& UAL::RCIPtr<T>::operator*() const 
{
  return *(m_counter->m_pointee); 
}

template<class T>
bool UAL::RCIPtr<T>::isValid() const 
{
  return m_counter->m_pointee != 0; 
}

#endif
