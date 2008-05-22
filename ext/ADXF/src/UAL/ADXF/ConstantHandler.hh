//# Library     : UAL
//# File        : UAL/ADXF/ConstantHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_CONSTANT_HANDLER_HH
#define UAL_ADXF_CONSTANT_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/BasicHandler.hh"

namespace UAL {

  /**
   * The constant handler
   */ 

  class ADXFConstantHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFConstantHandler();

    /** Destructor */
    ~ADXFConstantHandler();


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

    XMLCh* m_chName;
    XMLCh* m_chValue;
    

  };

}

#endif
