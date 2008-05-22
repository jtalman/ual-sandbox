//# Library     : UAL
//# File        : UAL/ADXF/ElementSetHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_ELEMENT_SET_HANDLER_HH
#define UAL_ADXF_ELEMENT_SET_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>
#include "xercesc/sax2/Attributes.hpp"

#include "SMF/PacGenElement.h"

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/BasicHandler.hh"

namespace UAL {

  /**
   * The basis SAX2 class of different handlers dealing with element sets
   */ 

  class ADXFElementSetHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFElementSetHandler();

    /** Destructor */
    ~ADXFElementSetHandler();

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

    void setGenElement(PacGenElement* genElement) { p_genElement = genElement; }

  protected:

    PacGenElement* p_genElement;

  private:


  };

}

#endif
