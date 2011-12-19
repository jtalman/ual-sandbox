// Library       : UAL
// File          : UAL/APF/PropagatorSequence.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_PROPAGATOR_SEQUENCE_HH
#define UAL_PROPAGATOR_SEQUENCE_HH

#include <list>
#include "UAL/APF/PropagatorNodePtr.hh"

namespace UAL {

  /** An iterator over a sequence of propagators. */
  typedef  std::list<UAL::PropagatorNodePtr>::iterator PropagatorIterator;

  /** An ordered collection of accelerator propagators.
   */

  class PropagatorSequence : public UAL::PropagatorNode {
    
  public:

    /** Constructor */
    PropagatorSequence();

    /** Copy Constructor */
    PropagatorSequence(const PropagatorSequence& rhs);

    /** Destructor */
    virtual ~PropagatorSequence();

    /** Copy operator */
    PropagatorSequence& operator=(const PropagatorSequence& rhs);

    const char* getType();

    /** Returns true */
    bool isSequence();

    /** Returns the first node of the accelerator sector associated with this propagator 
	(not implemented)
     */
    UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
	(not implemented)
     */
    UAL::AcceleratorNode& getBackAcceleratorNode();

    /** Defines the lattice elemements 
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
	(not implemented) 
    */
    void setLatticeElements(const AcceleratorNode& sequence, int i0, int i1, 
			    const AttributeSet& attSet);

    /** Propagates probe through the associated accelerator nodes */
    void propagate(UAL::Probe& probe);

    /** Returns a copy of this node */
    PropagatorNode* clone();

    /** Returns a sequence size */
    int size() const;

    /** Adds the propagator node */
    void add(PropagatorNodePtr& node);

    /** Returns an iterator pointing to first element. */
    PropagatorIterator begin();

    /** Returns an iterator pointing to one-past-last element. */
    PropagatorIterator end();


  private:

    std::list<UAL::PropagatorNodePtr> m_nodes;
    static UAL::AcceleratorNode s_emptyAccNode;

  private:

    void copy(const PropagatorSequence& rhs);

  };

}

#endif
