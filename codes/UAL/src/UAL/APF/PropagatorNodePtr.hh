// Library       : UAL
// File          : UAL/APF/PropagatorNodePtr.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_PROPAGATOR_NODE_PTR_HH
#define UAL_PROPAGATOR_NODE_PTR_HH

#include "UAL/Common/RCObject.hh"
#include "UAL/APF/PropagatorNode.hh"

namespace UAL {

  /** Smart pointer of the accelerator propagator node */

  class PropagatorNodePtr {

  public :

    /** Constructor */
    PropagatorNodePtr();

    /** Creates a smart pointer from a real one */    
    PropagatorNodePtr(PropagatorNode* realPtr);

    /** Copy constructor */
    PropagatorNodePtr(const PropagatorNodePtr& rhs);

    /** Destructor */
    ~PropagatorNodePtr();

    /** Copy operator */
    PropagatorNodePtr& operator=(const PropagatorNodePtr& rhs);

    /** Dereferences a smart pointer to return a real pointer */    
    PropagatorNode* operator->() const;

    /** Dereferences a smart pointer to return a real pointer */ 
    PropagatorNode* getPointer();  

    /** Dereferences a smart pointer */
    PropagatorNode& operator*() const;   

    /** Checks if the real pointer is defined. */    
    bool isValid() const;

  private:

    struct CountHolder : public RCObject {
      ~CountHolder() { delete m_pointee; }
      PropagatorNode* m_pointee;
    };

    CountHolder* m_counter;
    
    void init();

  };

}



#endif
