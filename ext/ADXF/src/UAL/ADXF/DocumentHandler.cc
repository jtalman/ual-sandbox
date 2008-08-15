
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/DocumentHandler.hh"
#include "UAL/ADXF/ADXFHandler.hh"


XERCES_CPP_NAMESPACE_USE

std::string UAL::ADXFDocumentHandler::s_adxfTag("adxf");

UAL::ADXFDocumentHandler::ADXFDocumentHandler()
{
}

void UAL::ADXFDocumentHandler::startElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname,
				    const   xercesc::Attributes&     attrs)
{
    char* tag = XMLString::transcode(localname);
    // cout << "ADXFDocumentHandler::startElement: saw tag: "<< tag << endl;
    if(s_adxfTag.compare(tag)) {
      std::cout << "ADXFDocumentHandler::startElement: I saw wrong tag: "<< tag << endl;
      exit(0);
    }

    m_adxfHandler.setParent(this);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(&m_adxfHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(&m_adxfHandler);

    XMLString::release(&tag);

    // m_adxfHandler.startElement(uri, localname, qname, attrs);
}

void UAL::ADXFDocumentHandler::endElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname)
{
    char* tag = XMLString::transcode(localname);
    // cout << "ADXFDocumentHandler::endElement: saw tag: "<< tag << endl;

    if(s_adxfTag.compare(tag)) {
      std::cout << "ADXFDocumentHandler::endElement: I saw wrong tag: "<< tag << endl;
      exit(0);
    }

    XMLString::release(&tag);
}

void UAL::ADXFDocumentHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFDocument: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

