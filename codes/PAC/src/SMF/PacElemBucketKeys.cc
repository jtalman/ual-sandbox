#include "SMF/PacElemBucketKeys.h"

// Singleton
PacElemBucketKeys* PacElemBucketKeys::_instance = 0;

// Constructor
PacElemBucketKeys::PacElemBucketKeys()
{
}

// Copy operator
PacElemBucketKeys& PacElemBucketKeys::operator=(const PacElemBucketKeys&)
{
  return *this;
}

// Return the singleton
PacElemBucketKeys* PacElemBucketKeys::instance()
{
  if(!_instance){ _instance = new PacElemBucketKeys(); }
  return _instance;
}

// Adds the PacKey object into the PacElemBucketKeys collection

int PacElemBucketKeys::insert(const  PacElemBucketKey& key)
{
  return _extent.insert_unique(key).second;
}

PacElemBucketKeyIterator PacElemBucketKeys::find(int index)
{
  return _extent.find(index);
}


PacElemBucketKeyIterator PacElemBucketKeys::end()
{
  return _extent.end();
}


PacElemBucketKeyIterator PacElemBucketKeys::begin()
{
  return _extent.begin();
}


int PacElemBucketKeys::size() const
{
  return _extent.size();
}
