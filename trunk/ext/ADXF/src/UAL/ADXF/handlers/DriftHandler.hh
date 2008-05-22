//# Library     : UAL
//# File        : UAL/ADXF/handlers/DriftHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_DRIFT_HANDLER_HH
#define UAL_ADXF_DRIFT_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler of the drift element 
   */ 

  class ADXFDriftHandler : public ADXFElementHandler
  {
  public:

    /** Constructor */
    ADXFDriftHandler();

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
