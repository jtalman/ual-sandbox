//# Library     : UAL
//# File        : UAL/ADXF/handlers/MfieldHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_MFIELD_HANDLER_HH
#define UAL_ADXF_MFIELD_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementSetHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler of the mfield attributes 
   */ 

  class ADXFMfieldHandler : public ADXFElementSetHandler
  {
  public:

    /** Constructor */
    ADXFMfieldHandler();

    /** Destructor */
    ~ADXFMfieldHandler();

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

    void addField(const   xercesc::Attributes& attrs); 


  protected:

    XMLCh* m_chML;  
    XMLCh* m_chA;  
    XMLCh* m_chB;  
 
  };

}

#endif
