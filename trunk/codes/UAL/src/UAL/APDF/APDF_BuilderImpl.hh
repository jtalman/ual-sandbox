// Library       : UAL
// File          : UAL/APDF/APDF_BuilderImpl.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_AP_BUILDER_IMPL_HH
#define UAL_AP_BUILDER_IMPL_HH

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#include "UAL/Common/AttributeSet.hh"
#include "UAL/SMF/AcceleratorNode.hh"
#include "UAL/APF/AcceleratorPropagator.hh"

#include "UAL/APDF/APDF_CreateElement.hh"

namespace UAL {

  /** A private part of the XML-based builder of the accelerator propagator
   */

  class APDF_BuilderImpl : public Object  {

  public:

    /** Constructor */
    APDF_BuilderImpl();

    /** Destructor */
    virtual ~APDF_BuilderImpl();

    static APDF_BuilderImpl& getInstance();

    /** Defines beam attributes  */
    void setBeamAttributes(const AttributeSet& ba);

    /** Parses via a file path or URL and returns AcceleratorPropagator */
    AcceleratorPropagator* parse(std::string& url);

  private:

    void setLattice(const std::string& latticeName);

  private:

    // Beam attributes 
    RCIPtr<AttributeSet> m_ba;

    // Lattice (will be moved to the Accelerator Propagator)
    AcceleratorNode*  m_lattice;


  private:

    xmlNodePtr getXmlNode(xmlNodePtr parentNode, const char* tag);

    AcceleratorPropagator*  createAP(xmlNodePtr propagatorNode);

    void createLinks(AcceleratorPropagator* ap, 
		     UAL::APDF_CreateElement& createElement);
    void addDefaultSectorLink(std::list<PropagatorNodePtr>& links,
			      UAL::APDF_LinkElement& defaultSectorLink,
			      UAL::AcceleratorPropagator* ap);
    void addDefaultTypeLink(std::list<PropagatorNodePtr>& links,
			    UAL::APDF_LinkElement& defaultTypeLink,
			    UAL::AcceleratorPropagator* ap);
    void makeElementLinks(std::list<PropagatorNodePtr>& elementLinks, 
			  UAL::APDF_CreateElement& createElement);
    void makeTypeLinks(std::list<PropagatorNodePtr>& typeLinks, 
		       UAL::APDF_CreateElement& createElement);


  private:

    static APDF_BuilderImpl* s_theInstance;

  };

}


#endif
