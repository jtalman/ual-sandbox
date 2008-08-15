
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/SectorsHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFSectorsHandler::ADXFSectorsHandler()
{
}

void UAL::ADXFSectorsHandler::startElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname,
				    const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSectorsHandler::stratElement: saw element: "<< message << endl;
    XMLString::release(&message);

    m_sectorHandler.setParent(this);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(&m_sectorHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(&m_sectorHandler);

    m_sectorHandler.startElement(uri, localname, qname, attrs);

}

void UAL::ADXFSectorsHandler::endElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSectorsHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFSectorsHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Sectors Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

