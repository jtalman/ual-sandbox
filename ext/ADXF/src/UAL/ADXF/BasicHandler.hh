//# Library     : UAL
//# File        : UAL/ADXF/BasicHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_BASIC_HANDLER_HH
#define UAL_ADXF_BASIC_HANDLER_HH

#include <map>

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"

namespace UAL {

  /**
   * The basic class of ADXF SAX2-based handlers
   */ 

  class ADXFBasicHandler : public xercesc::DefaultHandler
  {
  public:

    /** Constructor */
    ADXFBasicHandler();

    /** Destructor */
    virtual ~ADXFBasicHandler();

    void setParent(ADXFBasicHandler* parentHandler) { p_parentHandler = parentHandler; }

    /** Starts an element. */
    void startElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname,
        const   xercesc::Attributes&     attrs
    );

    /** Ends an element. */
    void endElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname
    );

    /** Reports a fatal XML parsing error. */
    void fatalError(const xercesc::SAXParseException&);

  protected:

    ADXFBasicHandler* p_parentHandler;

  };

}

#endif
