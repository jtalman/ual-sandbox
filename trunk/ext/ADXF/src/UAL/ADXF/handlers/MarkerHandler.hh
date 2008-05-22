//# Library     : UAL
//# File        : UAL/ADXF/handlers/MarkerHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_MARKER_HANDLER_HH
#define UAL_ADXF_MARKER_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler of the marker element 
   */ 

  class ADXFMarkerHandler : public ADXFElementHandler
  {
  public:

    /** Constructor */
    ADXFMarkerHandler();

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

  };

}

#endif
