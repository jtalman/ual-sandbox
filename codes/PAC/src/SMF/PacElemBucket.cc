// Library     : PAC
// File        : SMF/PacElembBucket.cc
// Copyright   : see Copyright file
// Description : Implementation of PacElembBucket.cc.
// Author      : Nikolay Malitsky

#include "SMF/PacElemBucket.h"

// Constructors

PacElemBucket::PacElemBucket(const PacElemBucketKey& k, int order)
  : _key(k.key()), _capacity(0), _size(0), _data(0)
{
  check(order, k);
  create(k.size()*(order+1), 0);
}

PacElemBucket::PacElemBucket(const PacElemBucket& bucket)
  : _key(bucket.key()), _capacity(0), _size(0), _data(0)
{
  create(bucket._size, bucket._data);
}

PacElemBucket& PacElemBucket::operator = (const PacElemBucket& bucket)
{
  check(bucket.key());
  create(bucket._size, bucket._data);

  return *this;
}

PacElemBucket& PacElemBucket::operator += (const PacElemBucket& bucket)
{
  check(bucket.key());
  extend(bucket._size);

  for(int i=0; i < bucket._size; i++) _data[i] += bucket._data[i];  

  return *this;
}

double PacElemBucket::operator [] (const PacElemAttribKey& k) const
{
  int order = _size/k.bucketKey().size() - 1;
  
  if(order < k.order()) { return 0.0; }
  return _data[k.index()];
}

PacElemBucket& PacElemBucket::operator -= (const PacElemBucket& bucket)
{
  check(bucket.key());
  extend(bucket._size);

  for(int i=0; i < bucket._size; i++) _data[i] -= bucket._data[i];

  return *this;
}

PacElemBucket& PacElemBucket::operator *= (double v)
{
  for(int i=0; i < _size; i++) _data[i] *= v;

  return *this;
}

PacElemBucket operator*(const PacElemAttribKey& key, double v)
{
  PacElemBucket bucket(key.bucketKey(), key.order());
  bucket._data[key.index()] = v;
  return bucket;
}

PacElemBucket operator*(double v, const PacElemAttribKey& key) 
{ 
  return key*v; 
}

int PacElemBucket::keySize() const
{
  return bucketKey().size();
}

int PacElemBucket::order() const
{
  return  _size/keySize() - 1;
}

void PacElemBucket::order(int value)
{
  int size = (value + 1)*keySize();
  if(size < 0) size = 0;

  if(size == _size) return;

  if(size < _size) _size = size;
  else extend(size);
}

void PacElemBucket::zero()
{
  for(int i=0; i < _size; i++) _data[i] = 0.0;
}

// Private methods

void PacElemBucket::create(int s, double* data)
{
  if(_size != s){

    if(_data) delete [] _data;

    _capacity = s;
    _size = s;
    _data = new double[s];

    if(!_data){
      std::string msg = "Error : PacElemBucket::create(size) : allocation failure \n";
      PacAllocError(msg).raise();
    }
  }

  if(data) { for(int i=0; i < _size; i++) _data[i] = data[i]; }
  else     { for(int i=0; i < _size; i++) _data[i] = 0.0; }

}

void PacElemBucket::extend(int s)
{
  if(_size >= s) return;

  int i;
  if(_capacity >= s){
    for(i=_size; i < s; i++) _data[i] = 0.0;
    _size = s;
  }

  int old_s = _size;

  if(old_s){
    double* tmp = new double[old_s];
    for(i=0; i < old_s; i++) tmp[i]   = _data[i];
    create(s, 0);
    for(i=0; i < old_s; i++) _data[i] = tmp[i];
    delete [] tmp;
  }
  else {
    create(s, 0);
  }

}

void PacElemBucket::check(int key) const 
{
  if(_key != key) { 
    std::string msg = "Error : PacElemBucket::check(int key) : _key != key \n";
    PacDomainError(msg).raise();
  }
}

void PacElemBucket::check(const PacElemAttribKey& k) const 
{
  check(k.bucketKey().key());

  int order = _size/k.bucketKey().size() - 1;
  
  if(order < k.order()) { 
    std::string msg  = "Error : PacElemBucket::check(const PacElemAttribKey& k) : ";
           msg += "bucket's order < k.order() \n";
    PacDomainError(msg).raise();
  }
}


void PacElemBucket::check(int order, const PacElemBucketKey& bucketKey) const
{
  if(order < 0) { 
    std::string msg = "Error : PacElemBucket::check(int order, const SmfElemBucketKey& bucketKey) : order < 0 \n";
    PacDomainError(msg).raise();
  }
  if(order > 0) { 
    if(!bucketKey.order()){
      std::string msg  = "Error : PacElemBucket::check(int order, const SmfElemBucketKey& bucketKey) ";
	     msg += ": order >  0 for bucketKey ";
      PacDomainError(msg + bucketKey.name()).raise();
    }
  }
}


