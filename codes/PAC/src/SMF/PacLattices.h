#ifndef PAC_SMF_LATTICES_H
#define PAC_SMF_LATTICES_H

#include "Templates/PacRbTree.h"
#include "SMF/PacSmfDef.h"
#include "SMF/PacLattice.h"

class PacLattices
{
public:

  // Return a pointer to the singleton
  static PacLattices* instance();

  typedef PacRbTree<string, PacLattice, PacNameOfLattice, PacLess<string> >::iterator iterator;

  // Adds the PacLattice object into the PacLattices  collection
  int insert(const PacLattice& e); 

  // Returns the iterator associated with the specified key. 
  iterator find(const string& key);

  void clean();
  int size() { return _extent.size(); }

  //  Returns the iterator associated with the collection end.
  iterator end();

  //  Returns the iterator associated with the collection begin.
  iterator begin();

protected:

  // Constructor
  PacLattices();

  // Copy operator
  PacLattices& operator=(const PacLattices&);
  
  // Singleton
  static PacLattices* _instance;

 PacRbTree<string, PacLattice, PacNameOfLattice, PacLess<string> > _extent;

};

typedef PacLattices::iterator PacLatticeIterator; 

#endif
