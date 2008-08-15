
#include <iostream>

#include "SMF/PacElemMultipole.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/HkickerHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFHkickerHandler::ADXFHkickerHandler()
{
  m_chKick  = new XMLCh[100];
  XMLString::transcode("kick", m_chKick, 99);
}

UAL::ADXFHkickerHandler::~ADXFHkickerHandler()
{
  XMLString::release(&m_chKick);
}

void UAL::ADXFHkickerHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacHkicker genElement(name);

    XMLString::release(&name);

    double l = addLength(genElement, attrs);
    // if(l == 0.0) return;

    addKick(genElement, l, attrs);
}

void UAL::ADXFHkickerHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFHkickerHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFHkickerHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFHkickerHandler::addKick(PacGenElement& genElement,
				      double l,
				       const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlKick  = attrs.getValue(m_chKick);
    if(xmlKick == 0) return;

    char* chKick  = XMLString::transcode(xmlKick);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chKick);
    double kick = muParser.Eval();

    XMLString::release(&chKick);

    if(kick == 0.0) return;

    PacElemMultipole mult(0);
    mult.kl(0) = kick;
    genElement.add(mult);

    // std::cout << "UAL::ADXFElementHandler::addKick " << kick << std::endl;
}


