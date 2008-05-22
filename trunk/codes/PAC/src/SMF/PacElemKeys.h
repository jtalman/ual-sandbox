#ifndef PAC_SMF_ELEM_KEYS_H
#define PAC_SMF_ELEM_KEYS_H

#include "Templates/PacRbTree.h"
#include "SMF/PacSmfDef.h"
#include "SMF/PacElemKey.h"

class PacElemKeys 
{
public:


  // Return a pointer to the singleton
  static PacElemKeys* instance();

  // Iterators of the PacElemKeys collection
  typedef PacRbTree<int, PacElemKey, PacKeyOfElemKey, PacLess<int> >::iterator iterator;

  // Adds the PacKey object into the PacElemKeys collection
  int insert(const PacElemKey& key); 

  // Adds the PacKey object into the PacElemKeys collection 
  // int insert_unique(const PacElemKey& key); 

  // Returns the iterator associated with the specified key. 
  iterator find(int index);

  //  Returns the iterator associated with the collection end.
  iterator end();

  //  Returns the iterator associated with the collection begin.
  iterator begin();

  //  Returns the size of collection
  int size() const;



protected:

  // Constructor
  PacElemKeys();

  // Copy operator
  PacElemKeys& operator=(const PacElemKeys&);
  
  // Singleton
  static PacElemKeys* _instance;

  PacRbTree<int, PacElemKey, PacKeyOfElemKey, PacLess<int> > _extent;

};

typedef PacElemKeys::iterator PacElemKeyIterator;

#endif
