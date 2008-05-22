// Library       : UAL
// File          : UAL/SMF/AcceleratorNodeFinder.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_ACCELERATOR_NODE_FINDER_HH
#define UAL_ACCELERATOR_NODE_FINDER_HH

#include <map>

#include "UAL/SMF/AcceleratorNode.hh"

namespace UAL { 

  /** Extent of accelerator nodes */

  class AcceleratorNodeFinder {

  public:
    
    /** An iterator over a collection of accelerator nodes. */
    typedef  std::map<std::string, AcceleratorNodePtr>::const_iterator Iterator;


  public:

    /** Returns the singleton */
    static AcceleratorNodeFinder& getInstance();

    /** Returns the empty node */
    static AcceleratorNode& getEmptyNode();

    void clean();
    int size() { return m_extent.size(); }

    /** Returns an iterator pointing to first element. */
    Iterator begin();

    /** Returns the iterator of the specified object */
    Iterator find(const std::string& id);

    /** Registers the accelerator node */
    void add(const AcceleratorNodePtr& node);

    /** Returns an iterator pointing to one-past-last element. */
    Iterator end();

  private:

    /** Singleton */
    static AcceleratorNodeFinder* s_theInstance;

    /** Empty Accelerator Node */
    static AcceleratorNode s_emptyAccNode;

    /** Extent of accelerator nodes */
    std::map<std::string, AcceleratorNodePtr> m_extent;

  private:

    /** Constructor */
    AcceleratorNodeFinder();


  };

}


#endif
