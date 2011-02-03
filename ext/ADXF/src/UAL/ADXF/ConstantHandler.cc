
#include <iostream>

#include "xercesc/sax2/Attributes.hpp"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ConstantHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFConstantHandler::ADXFConstantHandler()
{
  m_chName  = new XMLCh[100];
  XMLString::transcode("name", m_chName, 99);
  m_chValue = new XMLCh[100];
  XMLString::transcode("value", m_chValue, 99);
}

UAL::ADXFConstantHandler::~ADXFConstantHandler()
{
  XMLString::release(&m_chName);
  XMLString::release(&m_chValue);
}

void UAL::ADXFConstantHandler::startElement(const   XMLCh* const    uri,
					    const   XMLCh* const    localname,
					    const   XMLCh* const    qname,
					    const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFConstantHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    const XMLCh* xmlValue = attrs.getValue(m_chValue);

    char* name    = XMLString::transcode(xmlName);
    char* chValue = XMLString::transcode(xmlValue);

    // std::string strValue("1.2*3.14");
    // std::cout << "name = " << name << ", value = " << chValue << std::endl;

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    const mu::valmap_type& constants = muParser.GetConst();
    mu::valmap_type::const_iterator it = constants.find(name);
    if(it != constants.end()){
      std::cout << "Constant " << name << " has been already defined " << std::endl;
      exit(0);
    }

    muParser.SetExpr(chValue);
    double value = muParser.Eval();

    muParser.DefineConst(name, value);

    XMLString::release(&name);
    XMLString::release(&chValue);
    
}

void UAL::ADXFConstantHandler::endElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFConstantHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFConstantHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFConstantHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

