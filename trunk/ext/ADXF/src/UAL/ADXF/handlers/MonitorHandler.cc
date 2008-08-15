
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/MonitorHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFMonitorHandler::ADXFMonitorHandler()
{
}

void UAL::ADXFMonitorHandler::startElement(const   XMLCh* const    uri,
					   const   XMLCh* const    localname,
					   const   XMLCh* const    qname,
					   const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFDriftHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacMonitor genElement(name);

    XMLString::release(&name);

    addLength(genElement, attrs);

}

void UAL::ADXFMonitorHandler::endElement(const   XMLCh* const    uri,
					 const   XMLCh* const    localname,
					 const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFMonitorHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFMonitorHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFMonitorHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

