//# Library     : UAL
//# File        : UAL/ADXF/ADXFHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_HANDLER_HH
#define UAL_ADXF_HANDLER_HH

#include <map>

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/BasicHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler dealing with the ADXF tag
   */ 

  class ADXFHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFHandler();

    /** Destructor */
    ~ADXFHandler();

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

    std::map<std::string, ADXFBasicHandler*> m_handlers;

  };

}

#endif
