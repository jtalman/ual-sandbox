// Library       : UAL
// File          : UAL/SMF/AcceleratorNode.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_ACCELERATOR_NODE_HH
#define UAL_ACCELERATOR_NODE_HH

#include "UAL/Common/RCIPtr.hh"
#include "UAL/Common/Element.hh"
// #include "UAL/SMF/AcceleratorComponent.hh"

namespace UAL {

  /** A basis class of accelerator nodes, elements and sequences of elements */

  class AcceleratorNode : public Element {

  public:

    /** Constructor */
    AcceleratorNode();

    /** Destructor */
    virtual ~AcceleratorNode();

    /** Defines an accelerator component */
    // void setAcceleratorComponent(const AcceleratorComponentPtr& component); 

    /** Returns an accelerator component */
    // const AcceleratorComponentPtr& getAcceleratorComponent() const;    

    /** Returns the longitudinal position of this node */
    virtual double getPosition() const;

    /** Set the longitudinal position */
    virtual void setPosition(double at); 

    // Old interface

    /** Returns a type */
    virtual const std::string& getType() const;    

    /** Returns a name */
    virtual const std::string& getName() const;

    /** Returns a design name */   
    virtual const std::string& getDesignName() const; 

    /** Returns a length */
    virtual double getLength() const;

    /** Returns a number of children */   
    virtual int getNodeCount() const;

    /** Returns the specified child 
     * (return value will be replaced by AcceleratorNodePtr) 
     */   
    virtual AcceleratorNode* const getNodeAt(int indx) const;

    /** Returns a deep copy of this node */
    virtual AcceleratorNode* clone() const;

  protected:

    /** Accelerator component */
    // AcceleratorComponentPtr m_componentPtr;

  private:

    static std::string s_emptyName;

  };

  /** Smart pointer of the accelerator node */
  typedef RCIPtr<AcceleratorNode> AcceleratorNodePtr;

}


#endif
