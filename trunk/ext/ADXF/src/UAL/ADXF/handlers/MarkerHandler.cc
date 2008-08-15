
#include <iostream>
#include "xercesc/sax2/Attributes.hpp"

#include "SMF/PacGenElement.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/MarkerHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFMarkerHandler::ADXFMarkerHandler()
{
}

void UAL::ADXFMarkerHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFMarkerHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacMarker genElement(name);

    XMLString::release(&name);
}

void UAL::ADXFMarkerHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFMarkerHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFMarkerHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFMarkerHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

