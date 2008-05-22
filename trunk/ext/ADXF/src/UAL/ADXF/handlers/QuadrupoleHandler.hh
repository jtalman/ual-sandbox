//# Library     : UAL
//# File        : UAL/ADXF/handlers/QuadrupoleHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_QUADRUPOLE_HANDLER_HH
#define UAL_ADXF_QUADRUPOLE_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler of the quadrupole element 
   */ 

  class ADXFQuadrupoleHandler : public ADXFElementHandler
  {
  public:

    /** Constructor */
    ADXFQuadrupoleHandler();

    /** Constructor */
    ~ADXFQuadrupoleHandler();


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

    void addK1(PacGenElement& genElement, double l,
	       const   xercesc::Attributes& attrs); 

  protected:

    XMLCh* m_chK1;    

  };

}

#endif
