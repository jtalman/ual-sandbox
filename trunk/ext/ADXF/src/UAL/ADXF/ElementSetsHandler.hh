//# Library     : UAL
//# File        : UAL/ADXF/ElementsHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_ELEMENT_SETS_HANDLER_HH
#define UAL_ADXF_ELEMENT_SETS_HANDLER_HH

#include <map>

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>

#include "SMF/PacGenElement.h"

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementSetHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler dealing with the element attributes
   */ 

  class ADXFElementSetsHandler : public ADXFBasicHandler
  {
  public:

    static ADXFElementSetsHandler* getInstance();

    /** Destructor */
    ~ADXFElementSetsHandler();

    void startElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname,
        const   xercesc::Attributes&     attrs
    );

    void endElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname
    );

    void fatalError(const xercesc::SAXParseException&);

    void setGenElement(PacGenElement& genElement);

  protected:

    /** Constructor */
    ADXFElementSetsHandler();   

  protected:

    static ADXFElementSetsHandler* s_theInstance; 

  protected:

    PacGenElement* p_genElement;

  private:

    std::map<std::string, ADXFElementSetHandler*> m_handlers;

  };

}

#endif
