
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ADXFHandler.hh"
#include "UAL/ADXF/ConstantsHandler.hh"
#include "UAL/ADXF/ElementsHandler.hh"
#include "UAL/ADXF/SectorsHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFHandler::ADXFHandler()
{
  m_handlers["constants"] = new UAL::ADXFConstantsHandler();
  m_handlers["elements"] = new UAL::ADXFElementsHandler();
  m_handlers["sectors"] = new UAL::ADXFSectorsHandler();
}

UAL::ADXFHandler::~ADXFHandler()
{
  std::map<std::string, UAL::ADXFBasicHandler*>::iterator it; 
  for(it = m_handlers.begin(); it != m_handlers.end(); it++){
    if(it->second) delete it->second;
    it->second = 0;
  }
}


void UAL::ADXFHandler::startElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname,
				    const   xercesc::Attributes&     attrs)
{
    char* tag = XMLString::transcode(localname);
    // cout << "ADXFHandler::startElement: I saw tag: "<< tag << endl;
    std::map<std::string, UAL::ADXFBasicHandler*>::iterator it = m_handlers.find(tag);
    if(it == m_handlers.end()) {
      std::cout << "ADXFHandler::startElement: I saw wrong tag: "<< tag << endl;
      exit(0);
    }

    it->second->setParent(this);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(it->second);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(it->second);

    XMLString::release(&tag);

}

void UAL::ADXFHandler::endElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname)
{
    char* tag = XMLString::transcode(localname);
    // cout << "ADXFHandler::endElement: I saw tag: "<< tag << endl;
    XMLString::release(&tag);
}

void UAL::ADXFHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

