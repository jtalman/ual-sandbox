#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "SMF/PacLattices.h"

// Singleton
PacLattices* PacLattices::_instance = 0;

// Constructor
PacLattices::PacLattices()
{
}

// Copy operator
PacLattices& PacLattices::operator=(const PacLattices&)
{
  return *this;
}

// Return the singleton
PacLattices* PacLattices::instance()
{
  if(!_instance){ _instance = new PacLattices(); }
  return _instance;
}
 
int PacLattices::insert(const PacLattice& e)
{ 
  int result = _extent.insert_unique(e).second;

  if(result){
    UAL::AcceleratorNodePtr nodePtr(e.clone());
    UAL::AcceleratorNodeFinder::getInstance().add(nodePtr);
  }

  return result;
}

PacLatticeIterator PacLattices::find(const string& key)
{
  return _extent.find(key);
}

void PacLattices::clean()
{
  return _extent.clear();
}

PacLatticeIterator PacLattices::end()
{
  return _extent.end();
}


PacLatticeIterator PacLattices::begin()
{
  return _extent.begin();
}

