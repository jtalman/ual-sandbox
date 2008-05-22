// Library     : PAC
// File        : SMF/PacElemAttributes.h
// Copyright   : see Copyright file
// Description : Element attributes are presented as an associative array of PacElemBucket's, 
//               orthogonal collections of physical parameters (dipole, multipole field, etc.)
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_ATTRIBUTES
#define PAC_ELEM_ATTRIBUTES

#include "SMF/PacElemBucket.h"
#include "Templates/PacOrderedList.h"


struct PacKeyOfElemBucket
{
  const int& operator()(const PacElemBucket& x) const { return x.key(); }
};

class PacElemAttributes
{
public:

  // Constructors
 
  PacElemAttributes();
  PacElemAttributes(const PacElemAttributes& att);
  
  typedef PacOrderedList<int, PacElemBucket, PacKeyOfElemBucket, PacLess<int> >::iterator iterator;
  typedef PacOrderedList<int, PacElemBucket, PacKeyOfElemBucket, PacLess<int> >::const_iterator const_iterator;

  // Returns the iterator associated with the specified key. 
  iterator find(const int& index);

  //  Returns the iterator associated with the collection end.
  iterator end();

  //  Returns the iterator associated with the collection begin.
  iterator begin();

  //  Erase member of collection from begin to end.
  void erase( iterator begin, iterator end);

  //  Returns the size of collection
  int size() const;


  // "Standard Input Language"

  void set(const PacElemAttributes& att) { operator=(att); }
  void set(const PacElemBucket& bucket) { operator=(bucket);}

  void add(const PacElemAttributes& att) { operator+=(att); }
  void add(const PacElemBucket& bucket) { operator+=(bucket);} 

  double get(const PacElemAttribKey& key) const;

  void remove(const PacElemAttribKey& key);
  void remove() {_extent.erase(_extent.begin(), _extent.end()); } 

  // Interface

  PacElemAttributes& operator  = (const PacElemAttributes& att);
  PacElemAttributes& operator  = (const PacElemBucket& bucket);

  PacElemAttributes& operator += (const PacElemAttributes& att);
  PacElemAttributes& operator += (const PacElemBucket& bucket);

  PacElemAttributes& operator -= (const PacElemAttributes& att);
  PacElemAttributes& operator -= (const PacElemBucket& bucket);


  friend PacElemAttributes& operator , (const PacElemBucket& b1, const PacElemBucket& b2);
  friend PacElemAttributes& operator , (const PacElemAttributes& a, const PacElemBucket& b);
  friend PacElemAttributes& operator , (const PacElemBucket& b, const PacElemAttributes& a);

  // Interface to data

  PacElemBucket& operator[](const PacElemBucketKey& i) { return *(_extent.insert_unique(PacElemBucket(i)).first); };

 


 protected:
 
  PacOrderedList<int, PacElemBucket, PacKeyOfElemBucket, PacLess<int> > _extent;


};

typedef PacElemAttributes::iterator PacElemAttribIterator;

#endif
