//# Library     : UAL
//# File        : UAL/ADXF/ElementsHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_ELEMENTS_HANDLER_HH
#define UAL_ADXF_ELEMENTS_HANDLER_HH

#include <map>

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler dealing with the <elements> tag
   */ 

  class ADXFElementsHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFElementsHandler();

    /** Destructor */
    ~ADXFElementsHandler();

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

  private:

    std::map<std::string, ADXFElementHandler*> m_handlers;

  };

}

#endif
