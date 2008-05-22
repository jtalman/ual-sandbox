// Library     : UAL
// File        : UAL/Common/AttributeSet.hh
// Copyright   : see Copyright file
// Authors     : N.Malitsky & R.Talman

#ifndef UAL_ATTRIBUTE_SET_HH
#define UAL_ATTRIBUTE_SET_HH

#include <vector>
#include "UAL/Common/Object.hh"

namespace UAL {

/**
 * A root class of the UAL attribute sets.
 */

  class AttributeSet : public Object
  {

  public:

    /** Constructor */
    AttributeSet();

    /** Destructor */
    virtual ~AttributeSet();

    /** Returns the value of the specified attribute */
    virtual double getAttribute(const std::string& attrName) const;

    /** Defines the value of the specified attribute */
    virtual void setAttribute(const std::string& attrName, double value);

    /** Returns an unmodifiable array of names of simple attributes */
    virtual const std::vector<std::string>& getAttributeNames() const;

    /** Returns a deep copy */
    virtual AttributeSet* clone() const;

  private:

    // empty set of attribute names
    static std::vector<std::string> s_emptyVector;

  };


}

#endif
