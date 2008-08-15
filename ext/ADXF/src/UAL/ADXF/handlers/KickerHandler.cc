
#include <iostream>

#include "SMF/PacElemMultipole.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/KickerHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFKickerHandler::ADXFKickerHandler()
{
  m_chHkick  = new XMLCh[100];
  XMLString::transcode("hkick", m_chHkick, 99);
  m_chVkick  = new XMLCh[100];
  XMLString::transcode("vkick", m_chVkick, 99);
}

UAL::ADXFKickerHandler::~ADXFKickerHandler()
{
  XMLString::release(&m_chHkick);
  XMLString::release(&m_chVkick);
}

void UAL::ADXFKickerHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacKicker genElement(name);

    XMLString::release(&name);

    double l = addLength(genElement, attrs);
    // if(l == 0.0) return;

    addKicks(genElement, l, attrs);

    addAttributes(genElement);
}

void UAL::ADXFKickerHandler::endElement(const   XMLCh* const    uri,
					const   XMLCh* const    localname,
					const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFKickerHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFKickerHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFKickerHandler::addKicks(PacGenElement& genElement,
				      double l,
				      const   xercesc::Attributes& attrs)
{
    double hkick = getKickValue(m_chHkick, attrs);
    double vkick = getKickValue(m_chVkick, attrs);

    if(hkick == 0.0 && vkick == 0.0) return;

    PacElemMultipole mult(0);
    mult.kl(0)  = hkick;
    mult.ktl(0) = vkick;
    genElement.add(mult);

    // std::cout << "UAL::ADXFElementHandler::addKick " << kick << std::endl;
}

double UAL::ADXFKickerHandler::getKickValue(XMLCh* attKick,
					  const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlKick  = attrs.getValue(attKick);
    if(xmlKick == 0) return 0.0;

    char* chKick  = XMLString::transcode(xmlKick);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chKick);
    double kick = muParser.Eval();

    XMLString::release(&chKick);

    return kick;
}



