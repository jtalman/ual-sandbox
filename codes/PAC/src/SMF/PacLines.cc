#include "SMF/PacLines.h"

// Singleton
PacLines* PacLines::_instance = 0;

// Constructor
PacLines::PacLines()
{
}

// Copy operator
PacLines& PacLines::operator=(const PacLines&)
{
  return *this;
}

// Return the singleton
PacLines* PacLines::instance()
{
  if(!_instance){ _instance = new PacLines(); }
  return _instance;
}
 
int PacLines::insert(const PacLine& l)
{ 
return _extent.insert_unique(l).second;
}

PacLines::iterator PacLines::find(const string& key)
{
  return _extent.find(key);
}

void PacLines::clean()
{
  return _extent.clear();
}

PacLines::iterator PacLines::end()
{
  return _extent.end();
}


PacLines::iterator PacLines::begin()
{
  return _extent.begin();
}


int PacLines::size() const 
{
  return _extent.size();
}
