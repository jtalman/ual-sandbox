//# Library     : UAL
//# File        : UAL/ADXF/handlers/VkickerHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_VKICKER_HANDLER_HH
#define UAL_ADXF_VKICKER_HANDLER_HH

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>


#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementHandler.hh"

namespace UAL {

  /**
   * The SAX2 handler of the vkicker element 
   */ 

  class ADXFVkickerHandler : public ADXFElementHandler
  {
  public:

    /** Constructor */
    ADXFVkickerHandler();

    /** Constructor */
    ~ADXFVkickerHandler();


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

    void addKick(PacGenElement& genElement, double l,
		 const   xercesc::Attributes& attrs); 

  protected:

    XMLCh* m_chKick;    

  };

}

#endif
