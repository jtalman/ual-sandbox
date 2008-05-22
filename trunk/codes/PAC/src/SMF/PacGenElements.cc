#include "SMF/PacGenElements.h"

// Singleton
PacGenElements* PacGenElements::_instance = 0;

// Constructor
PacGenElements::PacGenElements()
{
}

// Copy operator
PacGenElements& PacGenElements::operator=(const PacGenElements&)
{
  return *this;
}

// Return the singleton
PacGenElements* PacGenElements::instance()
{
  if(!_instance){ _instance = new PacGenElements(); }
  return _instance;
}
 
int PacGenElements::insert(const PacGenElement& e)
{ 
return _extent.insert_unique(e).second;
}

PacGenElements::iterator PacGenElements::find(const string& n)
{
  return _extent.find(n);
}

void PacGenElements::clean()
{
  return _extent.clear();
}

PacGenElements::iterator PacGenElements::end()
{
  return _extent.end();
}


PacGenElements::iterator PacGenElements::begin()
{
  return _extent.begin();
}


int PacGenElements::size() const 
{
  return _extent.size();
}
