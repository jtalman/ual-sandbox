
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ElementSetsHandler.hh"

#include "UAL/ADXF/handlers/MfieldHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFElementSetsHandler* UAL::ADXFElementSetsHandler::s_theInstance = 0;

UAL::ADXFElementSetsHandler* UAL::ADXFElementSetsHandler::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new UAL::ADXFElementSetsHandler();
  }
  return s_theInstance;
}

UAL::ADXFElementSetsHandler::ADXFElementSetsHandler()
{
  p_genElement = 0;
  m_handlers["mfield"] = new UAL::ADXFMfieldHandler();
}

UAL::ADXFElementSetsHandler::~ADXFElementSetsHandler()
{
  std::map<std::string, UAL::ADXFElementSetHandler*>::iterator it; 
  for(it = m_handlers.begin(); it != m_handlers.end(); it++){
    if(it->second) delete it->second;
    it->second = 0;
  }
}

void UAL::ADXFElementSetsHandler::setGenElement(PacGenElement& genElement)
{
  PacGenElements* genElements =  PacGenElements::instance();
  PacGenElements::iterator it = genElements->find(genElement.name());
  p_genElement = &(*it);
  
}

void UAL::ADXFElementSetsHandler::startElement(const   XMLCh* const    uri,
					       const   XMLCh* const    localname,
					       const   XMLCh* const    qname,
					       const   xercesc::Attributes&     attrs)
{

    char* tag = XMLString::transcode(localname);
    // cout << "ADXFElementsHandler::startElement: I saw tag: "<< tag << endl;

    std::map<std::string, UAL::ADXFElementSetHandler*>::iterator it = m_handlers.find(tag);

    if(it == m_handlers.end()) {
      std::cout << "ADXFElementsSetHandler::startElement: I saw wrong tag: "<< tag << endl;
      UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
      UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
      XMLString::release(&tag);
      return;
    }

    cout << "ADXFElementSetsHandler::startElement: I saw tag: "<< tag << endl;

    it->second->setParent(this);
    it->second->setGenElement(p_genElement);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(it->second);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(it->second);

    XMLString::release(&tag);

    it->second->startElement(uri, localname, qname, attrs);
}

void UAL::ADXFElementSetsHandler::endElement(const   XMLCh* const    uri,
					     const   XMLCh* const    localname,
					     const   XMLCh* const    qname)
{

    char* tag = XMLString::transcode(localname);
    // cout << "ADXFElementSetsHandler::endElement: I saw tag: "<< tag << endl;

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);

    XMLString::release(&tag);

    p_parentHandler->endElement(uri, localname, qname);
}


void UAL::ADXFElementSetsHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Element Sets Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

