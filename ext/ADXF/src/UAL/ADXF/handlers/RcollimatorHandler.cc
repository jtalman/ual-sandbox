
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/RcollimatorHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFRcollimatorHandler::ADXFRcollimatorHandler()
{
}

void UAL::ADXFRcollimatorHandler::startElement(const   XMLCh* const    uri,
					       const   XMLCh* const    localname,
					       const   XMLCh* const    qname,
					       const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFDriftHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacRcollimator genElement(name);

    XMLString::release(&name);

    addLength(genElement, attrs);

}

void UAL::ADXFRcollimatorHandler::endElement(const   XMLCh* const    uri,
					     const   XMLCh* const    localname,
					     const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFDriftHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFRcollimatorHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFRcollimatorHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

