
#include <iostream>

#include "SMF/PacElemMultipole.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/QuadrupoleHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFQuadrupoleHandler::ADXFQuadrupoleHandler()
{
  m_chK1  = new XMLCh[100];
  XMLString::transcode("k1", m_chK1, 99);
}

UAL::ADXFQuadrupoleHandler::~ADXFQuadrupoleHandler()
{
  XMLString::release(&m_chK1);
}

void UAL::ADXFQuadrupoleHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacQuadrupole genElement(name);

    XMLString::release(&name);

    double l = addLength(genElement, attrs);
    if(l == 0.0) return;

    addK1(genElement, l, attrs);
}

void UAL::ADXFQuadrupoleHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFQuadrupoleHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFQuadrupoleHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFQuadrupoleHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFQuadrupoleHandler::addK1(PacGenElement& genElement,
				       double l,
				       const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlK1  = attrs.getValue(m_chK1);
    if(xmlK1 == 0) return;

    char* chK1  = XMLString::transcode(xmlK1);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chK1);
    double k1 = muParser.Eval();

    XMLString::release(&chK1);

    if(k1 == 0.0) return;

    PacElemMultipole mult(1);
    mult.kl(1) = k1*l;
    genElement.add(mult);

    // std::cout << "UAL::ADXFElementHandler::addK1 " << k1 << std::endl;
}


