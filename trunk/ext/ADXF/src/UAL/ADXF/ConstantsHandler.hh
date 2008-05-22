//# Library     : UAL
//# File        : UAL/ADXF/ConstantsHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_CONSTANTS_HANDLER_HH
#define UAL_ADXF_CONSTANTS_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ConstantHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler dealing with the <constants> tag
   */ 

  class ADXFConstantsHandler : public ADXFBasicHandler
  {
  public:

    ADXFConstantsHandler();

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

  protected:

    ADXFConstantHandler m_constHandler;

  };

}

#endif
