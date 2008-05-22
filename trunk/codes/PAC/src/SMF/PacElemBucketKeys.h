#ifndef PAC_SMF_ELEM_BUCKET_KEYS_H
#define PAC_SMF_ELEM_BUCKET_KEYS_H

#include "Templates/PacRbTree.h"
#include "SMF/PacSmfDef.h"
#include "SMF/PacElemBucketKey.h"

class PacElemBucketKeys
{
public:

  // Return a pointer to the singleton
  static PacElemBucketKeys* instance();

  typedef PacRbTree<int, PacElemBucketKey, PacKeyOfElemBucketKey, PacLess<int> >::iterator iterator;  

// Adds the PacKey object into the PacElemBucketKeys collection

  int insert(const PacElemBucketKey& key); 

  // Returns the iterator associated with the specified key. 
  iterator find(int index);

  //  Returns the iterator associated with the collection end.
  iterator end();

  //  Returns the iterator associated with the collection begin.
  iterator begin();

  //  Returns the iterator associated with the collection begin.
  int size() const;

protected:

  // Constructor
  PacElemBucketKeys();

  // Copy operator
  PacElemBucketKeys& operator=(const PacElemBucketKeys&);
  
  // Singleton
  static PacElemBucketKeys* _instance;

  PacRbTree<int, PacElemBucketKey, PacKeyOfElemBucketKey, PacLess<int> > _extent;


};

typedef PacElemBucketKeys::iterator PacElemBucketKeyIterator; 

#endif
