// Library       : UAL
// File          : UAL/APF/AcceleratorPropagator.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_ACCELERATOR_PROPAGATOR_HH
#define UAL_ACCELERATOR_PROPAGATOR_HH

#include "UAL/APF/PropagatorSequence.hh"

namespace UAL {

  /** A hierarchical tree of accelerator propagator nodes.
   */

  class AcceleratorPropagator : public Algorithm {

  public:

    /** Constructor */
    AcceleratorPropagator();

    /** Destructor */
    virtual ~AcceleratorPropagator();
    
    /** Defines the name */
    void setName(const std::string& name);

    /** Returns the name */
    const std::string& getName() const;    

    /** Returns the root node of this accelerator propagator*/
    UAL::PropagatorSequence& getRootNode();

    /** Returns a collection of propagators selected by associated element names */
    std::list<PropagatorNodePtr> getNodesByName(const std::string& pattern);

    /** Propages probe through the accelerator */
    void propagate(UAL::Probe& probe);

  private:

    std::string m_name;

    PropagatorSequence m_rootNode;

  };

}


#endif
