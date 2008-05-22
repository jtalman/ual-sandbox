// Library       : UAL
// File          : UAL/APDF/APDF_Builder.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_AP_BUILDER_HH
#define UAL_AP_BUILDER_HH

#include "UAL/Common/AttributeSet.hh"
#include "UAL/SMF/AcceleratorNode.hh"
#include "UAL/APF/AcceleratorPropagator.hh"

namespace UAL {

  /** An XML-based builder of the accelerator propagator, a hierarchical tree 
   * of accelerator propagator nodes.
   */

  class APDF_Builder : public Object  {

  public:

    /** Constructor */
    APDF_Builder();

    /** Destructor */
    virtual ~APDF_Builder();

    /** Defines beam attributes  */
    void setBeamAttributes(const AttributeSet& ba);

    /** Parses via a file path or URL and returns AcceleratorPropagator */
    AcceleratorPropagator* parse(std::string& url);
  
  };  

}

#endif
