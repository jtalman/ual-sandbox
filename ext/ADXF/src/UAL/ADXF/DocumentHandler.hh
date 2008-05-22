//# Library     : UAL
//# File        : UAL/ADXF/DocumentHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_DOCUMENT_HANDLER_HH
#define UAL_ADXF_DOCUMENT_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ADXFHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler dealing with the ADXF Document
   */ 

  class ADXFDocumentHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFDocumentHandler();


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

    static std::string s_adxfTag;
    ADXFHandler m_adxfHandler;

  };

}

#endif
