//# Library     : UAL
//# File        : UAL/ADXF/ElementHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_ELEMENT_HANDLER_HH
#define UAL_ADXF_ELEMENT_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>
#include "xercesc/sax2/Attributes.hpp"

#include "SMF/PacGenElement.h"

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/BasicHandler.hh"

namespace UAL {

  /**
   * The basis SAX2 class of different handlers dealing with elements 
   */ 

  class ADXFElementHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFElementHandler();

    /** Destructor */
    ~ADXFElementHandler();

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

    double addLength(PacGenElement& genElement, 
		     const   xercesc::Attributes& attrs);

    void addAttributes(PacGenElement& genElement);

  protected:

    XMLCh* m_chName;
    XMLCh* m_chL;

  };

}

#endif
