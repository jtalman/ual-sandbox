// Library     : PAC
// File        : SMF/PacElemAttribute.cc
// Copyright   : see Copyright file
// Description : Implementation of element attributes
// Author      : Nikolay Malitsky

#include "SMF/PacElemAttributes.h"

static PacElemAttributes __smfElemAttributes;

PacElemAttributes::PacElemAttributes()
 : _extent(PacLess<int>())           
{
}
 
PacElemAttributes::PacElemAttributes(const PacElemAttributes& att)
 : _extent(PacLess<int>())    
{

  const_iterator i;                             
  for(i =att._extent.begin(); i != att._extent.end(); i++) _extent.insert_unique(*i);
}

// "Standard Input Language"

double PacElemAttributes::get(const PacElemAttribKey& key) const
{
  const_iterator it = _extent.find(key.bucketKey().key());
  if(it == _extent.end()) return 0;

  return (*it)[key];
}

void PacElemAttributes::remove(const PacElemAttribKey& key)
{
  iterator it = _extent.find(key.bucketKey().key());
  if(it == _extent.end()) return;

  (*it)[key] = 0.0;
}


// Interface

PacElemAttributes&  PacElemAttributes::operator=(const PacElemAttributes& att)
{
  _extent.erase(_extent.begin(), _extent.end());

  const_iterator i;
  for(i = att._extent.begin(); i != att._extent.end(); i++) _extent.insert_unique(*i);

  return *this;
}

PacElemAttributes&  PacElemAttributes::operator=(const PacElemBucket& bucket)
{
 _extent.erase(_extent.begin(),_extent.end());
  _extent.insert_unique(bucket);
  return *this;
}

PacElemAttributes&  PacElemAttributes::operator+=(const PacElemAttributes& att)
{
  const_iterator i;                          
  iterator j;
  for(i = att._extent.begin(); i != att._extent.end(); i++){
    j =_extent.find((*i).key());
    if(j == _extent.end()) _extent.insert_unique(*i);
    else  (*j) += (*i) ;
  }
  return *this;
}

PacElemAttributes&  PacElemAttributes::operator+=(const PacElemBucket& bucket)
{
  iterator j = _extent.find(bucket.key());

  if(j == _extent.end()) _extent.insert_unique(bucket);
  else           (*j) += bucket ; 

  return *this;
}


PacElemAttributes& PacElemAttributes::operator-=(const PacElemAttributes& att)
{
  const_iterator i;                                    
  iterator j;
  for(i = att._extent.begin(); i != att._extent.end(); i++){
    j = _extent.find((*i).key());
    if(j == _extent.end()) *(_extent.insert_unique(*i).first) *= -1;
    else           (*j) -= (*i) ;
  }
  return *this;
}

PacElemAttributes& PacElemAttributes::operator-=(const PacElemBucket& bucket)
{
  iterator j = _extent.find(bucket.key());

  if(j == _extent.end()) *(_extent.insert_unique(bucket).first) *= -1;
  else           (*j) -= bucket ;

  return *this;
}

PacElemAttributes& operator , (const PacElemBucket& b1, const PacElemBucket& b2)
{
  __smfElemAttributes  = b1;
  __smfElemAttributes += b2;
  return __smfElemAttributes;
}

PacElemAttributes& operator , (const PacElemBucket& b, const PacElemAttributes& a)
{
  if(&a != &__smfElemAttributes){
    __smfElemAttributes = a;
  }
  __smfElemAttributes  += b;

  return __smfElemAttributes;
}

PacElemAttributes& operator , (const PacElemAttributes& a, const PacElemBucket& b)
{
  return operator,(b, a);
}


PacElemAttributes::iterator PacElemAttributes::begin()  
{
 return _extent.begin();
}

PacElemAttributes::iterator PacElemAttributes::find(const int& index)
{
 return _extent.find(index);
}

PacElemAttributes::iterator PacElemAttributes::end()
{
 return _extent.end();
}


void PacElemAttributes::erase(PacElemAttributes::iterator begin, PacElemAttributes::iterator end)
{
 _extent.erase (begin, end);
}


int PacElemAttributes::size() const 
{
  return _extent.size();
}

