// Library     : PAC
// File        : SMF/PacElembBucket.h
// Copyright   : see Copyright file
// Description : PacElemBucket - orthogonal collection of physical 
//               parameters (dipole, multipole field, etc.)
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_BUCKET_H
#define PAC_ELEM_BUCKET_H

#include "SMF/PacElemBucketKey.h"

class PacElemBucket;
PacElemBucket operator*(const PacElemAttribKey& key, double v);
PacElemBucket operator*(double v, const PacElemAttribKey& key);

class PacElemBucket
{

public:

  // Constructors & destructor

  PacElemBucket() : _key(0), _capacity(0), _size(0), _data(0) {}
  PacElemBucket(const PacElemBucketKey& key, int order = 0);
  PacElemBucket(const PacElemBucket& bucket);
  virtual ~PacElemBucket() { if(_data) delete [] _data; }

  // Keys

  const int& key() const   { return _key; }
  const PacElemBucketKey& bucketKey() const { return *pacBend(_key); }

  virtual int keySize() const;

  // Data

  // Return a vector size
  int size() const  { return _size;}

  // Return data
  double* data()    { return _data; } 

  // Return order
  virtual int order() const;
 
  // Set order
  virtual void order(int order);

  double  operator[](int index) const { return _data[index]; }
  double& operator[](int index) { return _data[index]; }

  double  operator[](const PacElemAttribKey& k) const;
  double& operator[](const PacElemAttribKey& k) { check(k); return _data[k.index()]; }

  // Operators

  PacElemBucket& operator  = (const PacElemBucket& bucket);

  PacElemBucket& operator += (const PacElemBucket& bucket);
  PacElemBucket& operator -= (const PacElemBucket& bucket);

  PacElemBucket& operator *= (double v);
  PacElemBucket& operator /= (double v) { return operator*=(1/v); }

  friend PacElemBucket operator*(const PacElemAttribKey& key, double v);
  friend PacElemBucket operator*(double v, const PacElemAttribKey& key);

  void zero();

protected:

  int _key;

  int _capacity;

  int _size;
  double* _data;

private:

  void check(int key) const ;
  void check(const PacElemAttribKey& key) const ;
  void check(int order, const PacElemBucketKey& bucketKey) const;

  void create(int size, double* data);
  void extend(int size);

};

#endif
