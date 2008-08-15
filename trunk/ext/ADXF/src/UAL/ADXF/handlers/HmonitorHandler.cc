
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/HmonitorHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFHmonitorHandler::ADXFHmonitorHandler()
{
}

void UAL::ADXFHmonitorHandler::startElement(const   XMLCh* const    uri,
					    const   XMLCh* const    localname,
					    const   XMLCh* const    qname,
					    const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFDriftHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacHmonitor genElement(name);

    XMLString::release(&name);

    addLength(genElement, attrs);

}

void UAL::ADXFHmonitorHandler::endElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFDriftHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFHmonitorHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFHmonitorHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

