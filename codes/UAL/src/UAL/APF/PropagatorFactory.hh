// Library       : UAL
// File          : UAL/APF/PropagatorFactory.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_PROPAGATOR_FACTORY_HH
#define UAL_PROPAGATOR_FACTORY_HH

#include <map>
#include "UAL/APF/PropagatorNodePtr.hh"

namespace UAL {

  /** A factory of registered propagators.
   */

  class PropagatorFactory   {

  public:
    
      /** An iterator over a collection of propagators. */
    typedef  std::map<std::string, PropagatorNodePtr>::const_iterator Iterator;

  public:

    /** Returns the only instance of this class*/
    static PropagatorFactory& getInstance();

    PropagatorNode* createPropagator(const std::string& classname);
    
    /** Registers the specified propagator */
    void add(const std::string& classname, const PropagatorNodePtr& ptr);

    /** Returns an iterator pointing to first element. */
    Iterator begin();

    /** Returns the iterator of the specified object */
    Iterator find(const std::string& classname);

    /** Returns an iterator pointing to one-past-last element. */
    Iterator end();
 
  private:

    // Singleton
    static PropagatorFactory* s_theInstance;

    // Registry
    std::map<std::string, PropagatorNodePtr > m_registry;

  private:

    // Constructor 
    PropagatorFactory();

  };

}


#endif
