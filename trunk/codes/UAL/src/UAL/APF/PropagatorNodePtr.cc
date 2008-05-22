// Library       : UAL
// File          : UAL/APF/PropagatorNodePtr.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream>
#include "UAL/APF/PropagatorNodePtr.hh"

void UAL::PropagatorNodePtr::init()
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


UAL::PropagatorNodePtr::PropagatorNodePtr()
  : m_counter(new UAL::PropagatorNodePtr::CountHolder())
{
  m_counter->m_pointee = 0;
  init();
}

UAL::PropagatorNodePtr::PropagatorNodePtr(PropagatorNode* realPtr)
  : m_counter(new UAL::PropagatorNodePtr::CountHolder())
{
  m_counter->m_pointee = realPtr;
  init();
}

UAL::PropagatorNodePtr::PropagatorNodePtr(const UAL::PropagatorNodePtr& rhs)
  : m_counter(rhs.m_counter)
{
  init();
}

UAL::PropagatorNodePtr::~PropagatorNodePtr()
{
  if(m_counter != 0 && m_counter->removeReference() <= 0) delete m_counter;
}

UAL::PropagatorNodePtr& UAL::PropagatorNodePtr::operator=(const UAL::PropagatorNodePtr& rhs)
{
  if(m_counter != rhs.m_counter) {
    if(m_counter != 0 && m_counter->removeReference() <= 0) delete m_counter;
    m_counter = rhs.m_counter;
    init();
  }
  return *this;
}

UAL::PropagatorNode* UAL::PropagatorNodePtr::operator->() const 
{
  return m_counter->m_pointee; 
}

UAL::PropagatorNode* UAL::PropagatorNodePtr::getPointer() 
{
  return m_counter->m_pointee; 
}

UAL::PropagatorNode& UAL::PropagatorNodePtr::operator*() const 
{
  return *(m_counter->m_pointee); 
}

bool UAL::PropagatorNodePtr::isValid() const 
{
  return m_counter->m_pointee != 0; 
}
