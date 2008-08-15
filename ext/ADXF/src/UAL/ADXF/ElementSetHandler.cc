
#include <iostream>

#include "SMF/PacElemLength.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ElementSetHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFElementSetHandler::ADXFElementSetHandler()
{
  p_genElement = 0;
}

UAL::ADXFElementSetHandler::~ADXFElementSetHandler()
{
}

void UAL::ADXFElementSetHandler::startElement(const   XMLCh* const    uri,
					      const   XMLCh* const    localname,
					      const   XMLCh* const    qname,
					      const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFElementHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);
}

void UAL::ADXFElementSetHandler::endElement(const   XMLCh* const    uri,
					    const   XMLCh* const    localname,
					    const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFElementHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFElementSetHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Element Set Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}


