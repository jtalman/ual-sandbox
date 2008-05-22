#ifndef PAC_SMF_GEN_ELEMENTS_H
#define PAC_SMF_GEN_ELEMENTS_H

#include "Templates/PacRbTree.h"
#include "SMF/PacSmfDef.h"
#include "SMF/PacGenElement.h"

class PacGenElements
{
public:

  // Return a pointer to the singleton
  static PacGenElements* instance();
 
 typedef PacRbTree<string, PacGenElement, PacNameOfGenElement, PacLess<string> >::iterator iterator;

  // Adds the PacGenElement object into the PacGenElements collection
  int insert(const PacGenElement& e); 

  // Returns the iterator associated with the specified key. 
  iterator find(const string& n);

  void clean();
  int size() { return _extent.size(); }

  //  Returns the iterator associated with the collection end.
  iterator end();

  //  Returns the iterator associated with the collection begin.
  iterator begin();

  //  Returns the size of collection
  int size() const;

protected:

  // Constructor
  PacGenElements();

  // Copy operator
  PacGenElements& operator=(const PacGenElements&);
  
  // Singleton
  static PacGenElements* _instance;
 
  PacRbTree<string, PacGenElement, PacNameOfGenElement, PacLess<string> > _extent;

};

typedef PacGenElements::iterator PacGenElemIterator; 

#endif
