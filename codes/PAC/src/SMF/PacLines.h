#ifndef PAC_SMF_LINES_H
#define PAC_SMF_LINES_H

#include "Templates/PacRbTree.h"
#include "SMF/PacSmfDef.h"
#include "SMF/PacLine.h"

class PacLines
{
public:

  // Return a pointer to the singleton
  static PacLines* instance();

  typedef PacRbTree<string, PacLine, PacNameOfLine, PacLess<string> >::iterator iterator;

  // Adds the PacLine object into the PacLines collection
  int insert(const PacLine& l); 

  // Returns the iterator associated with the specified key. 
  iterator find(const string& key);

  void clean();

  //  Returns the iterator associated with the collection end.
  iterator end();

 //  Returns the iterator associated with the collection begin.
  iterator begin();

  //  Returns the size of collection
  int size() const;

protected:

  // Constructor
  PacLines();

  // Copy operator
  PacLines& operator=(const PacLines&);
  
  // Singleton
  static PacLines* _instance;

  PacRbTree<string, PacLine, PacNameOfLine, PacLess<string> > _extent;

};

typedef PacLines::iterator PacLineIterator; 

#endif
