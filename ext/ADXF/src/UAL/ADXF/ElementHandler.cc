
#include <iostream>

#include "SMF/PacElemLength.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ElementHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"
#include "UAL/ADXF/ElementSetsHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFElementHandler::ADXFElementHandler()
{
  m_chName  = new XMLCh[100];
  XMLString::transcode("name", m_chName, 99);
  m_chL  = new XMLCh[100];
  XMLString::transcode("l", m_chL, 99);
}

UAL::ADXFElementHandler::~ADXFElementHandler()
{
  XMLString::release(&m_chName);
  XMLString::release(&m_chL);
}

void UAL::ADXFElementHandler::startElement(const   XMLCh* const    uri,
					   const   XMLCh* const    localname,
					   const   XMLCh* const    qname,
					   const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFElementHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);
}

void UAL::ADXFElementHandler::endElement(const   XMLCh* const    uri,
					   const   XMLCh* const    localname,
					   const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFElementHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFElementHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Element Handler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFElementHandler::addAttributes(PacGenElement& genElement)
{
  UAL::ADXFElementSetsHandler* setsHandler = UAL::ADXFElementSetsHandler::getInstance();
  setsHandler->setGenElement(genElement);
  setsHandler->setParent(this);

  UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(setsHandler);
  UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(setsHandler);
} 

double UAL::ADXFElementHandler::addLength(PacGenElement& genElement, 
					const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlL  = attrs.getValue(m_chL);
    if(xmlL == 0) return 0.0;

    char* chL  = XMLString::transcode(xmlL);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chL);
    double l = muParser.Eval();

    XMLString::release(&chL);

    if(l == 0.0) return l;

    PacElemLength length;
    length.l(l);
    genElement.add(length);

    // std::cout << "UAL::ADXFElementHandler::addLength " << l << std::endl;

    return l;
}

