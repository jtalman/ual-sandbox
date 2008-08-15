
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ConstantsHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFConstantsHandler::ADXFConstantsHandler()
{
}

void UAL::ADXFConstantsHandler::startElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname,
				    const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFConstantsHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    m_constHandler.setParent(this);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(&m_constHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(&m_constHandler);

    m_constHandler.startElement(uri, localname, qname, attrs);
}

void UAL::ADXFConstantsHandler::endElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFConstantsHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}


void UAL::ADXFConstantsHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Constants Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

