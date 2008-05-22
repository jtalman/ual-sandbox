// Library     : UAL
// File        : UAL/Common/RCObject.cc
// Copyright   : see Copyright file
// Authors     : implemented after Scott Meyers' approach ("More effective C++") 

#include "UAL/Common/RCObject.hh"

UAL::RCObject::RCObject() : refCount(0), shareable(true) 
{
}

UAL::RCObject::RCObject(const RCObject&) : refCount(0), shareable(true) 
{
}

UAL::RCObject& UAL::RCObject::operator=(const RCObject&)
{
  return *this;
}

UAL::RCObject::~RCObject()
{
}

int UAL::RCObject::addReference() 
{
  return ++refCount;
}

int UAL::RCObject::removeReference()
{
  return --refCount;
}

void UAL::RCObject::markUnshareable()
{
  shareable = false;
}

bool UAL::RCObject::isShareable() const 
{
  return shareable;
}

bool UAL::RCObject::isShared() const 
{
  return refCount > 1;
}
