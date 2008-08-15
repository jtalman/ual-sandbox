
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ElementsHandler.hh"

#include "UAL/ADXF/handlers/MarkerHandler.hh"
#include "UAL/ADXF/handlers/DriftHandler.hh"
#include "UAL/ADXF/handlers/SbendHandler.hh"
#include "UAL/ADXF/handlers/QuadrupoleHandler.hh"
#include "UAL/ADXF/handlers/SextupoleHandler.hh"
#include "UAL/ADXF/handlers/MonitorHandler.hh"

#include "UAL/ADXF/handlers/MultipoleHandler.hh"
#include "UAL/ADXF/handlers/InstrumentHandler.hh"
#include "UAL/ADXF/handlers/VmonitorHandler.hh"
#include "UAL/ADXF/handlers/HmonitorHandler.hh"
#include "UAL/ADXF/handlers/RcollimatorHandler.hh"
#include "UAL/ADXF/handlers/HkickerHandler.hh"
#include "UAL/ADXF/handlers/VkickerHandler.hh"
#include "UAL/ADXF/handlers/KickerHandler.hh"
#include "UAL/ADXF/handlers/SolenoidHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFElementsHandler::ADXFElementsHandler()
{
  m_handlers["element"] = new UAL::ADXFElementHandler();
  m_handlers["drift"] = new UAL::ADXFDriftHandler();
  m_handlers["hkicker"] = new UAL::ADXFHkickerHandler();
  m_handlers["hmonitor"] = new UAL::ADXFHmonitorHandler();
  m_handlers["instrument"] = new UAL::ADXFInstrumentHandler();
  m_handlers["kicker"] = new UAL::ADXFKickerHandler();
  m_handlers["marker"] = new UAL::ADXFMarkerHandler();
  m_handlers["monitor"] = new UAL::ADXFMonitorHandler();
  m_handlers["multipole"] = new UAL::ADXFMultipoleHandler();
  m_handlers["quadrupole"] = new UAL::ADXFQuadrupoleHandler();
  m_handlers["rcollimator"] = new UAL::ADXFRcollimatorHandler();
  m_handlers["rfcavity"] = new UAL::ADXFElementHandler();
  m_handlers["sbend"] = new UAL::ADXFSbendHandler();
  m_handlers["sextupole"] = new UAL::ADXFSextupoleHandler();
  m_handlers["solenoid"] = new UAL::ADXFSolenoidHandler();
  m_handlers["vkicker"] = new UAL::ADXFVkickerHandler();
  m_handlers["vmonitor"] = new UAL::ADXFVmonitorHandler();
}

UAL::ADXFElementsHandler::~ADXFElementsHandler()
{
  std::map<std::string, UAL::ADXFElementHandler*>::iterator it; 
  for(it = m_handlers.begin(); it != m_handlers.end(); it++){
    if(it->second) delete it->second;
    it->second = 0;
  }
}


void UAL::ADXFElementsHandler::startElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname,
				    const   xercesc::Attributes&     attrs)
{

    char* tag = XMLString::transcode(localname);
    // cout << "ADXFElementsHandler::startElement: I saw tag: "<< tag << endl;

    std::map<std::string, UAL::ADXFElementHandler*>::iterator it = m_handlers.find(tag);

    if(it == m_handlers.end()) {
      std::cout << "ADXFElementsHandler::startElement: I saw wrong tag: "<< tag << endl;
      exit(0);
    }

    it->second->setParent(this);

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(it->second);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(it->second);

    XMLString::release(&tag);

    it->second->startElement(uri, localname, qname, attrs);
}

void UAL::ADXFElementsHandler::endElement(const   XMLCh* const    uri,
				    const   XMLCh* const    localname,
				    const   XMLCh* const    qname)
{

    char* tag = XMLString::transcode(localname);
    // cout << "ADXFElementsHandler::endElement: I saw tag: "<< tag << endl;

    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);

    XMLString::release(&tag);
}


void UAL::ADXFElementsHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Elements Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

