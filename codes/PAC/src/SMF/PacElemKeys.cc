#include "SMF/PacElemKeys.h"

// Singleton
PacElemKeys* PacElemKeys::_instance = 0;

// Constructor
PacElemKeys::PacElemKeys()
{
}

// Copy operator
PacElemKeys& PacElemKeys::operator=(const PacElemKeys&)
{
  return *this;
}

// Return the singleton
PacElemKeys* PacElemKeys::instance()
{
  if(!_instance){ _instance = new PacElemKeys(); }
  return _instance;
}

int PacElemKeys::insert(const PacElemKey& key)
{
  return _extent.insert_unique(key).second;
}

PacElemKeyIterator PacElemKeys::find(int index)
{
  return _extent.find(index);
}


PacElemKeyIterator PacElemKeys::end()
{
  return _extent.end();
}


PacElemKeyIterator PacElemKeys::begin() 
{
  return _extent.begin();
}


int PacElemKeys::size() const  
{
  return _extent.size();
}
