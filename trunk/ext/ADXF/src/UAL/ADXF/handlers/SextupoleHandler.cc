
#include <iostream>

#include "SMF/PacElemMultipole.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/SextupoleHandler.hh"

#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFSextupoleHandler::ADXFSextupoleHandler()
{
  m_chK2  = new XMLCh[100];
  XMLString::transcode("k2", m_chK2, 99);
}

UAL::ADXFSextupoleHandler::~ADXFSextupoleHandler()
{
  XMLString::release(&m_chK2);
}

void UAL::ADXFSextupoleHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSextupoleHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacSextupole genElement(name);

    XMLString::release(&name);

    double l = addLength(genElement, attrs);
    if(l == 0.0) return;

    addK2(genElement, l, attrs);
}

void UAL::ADXFSextupoleHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSextupoleHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFSextupoleHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFQuadrupoleHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFSextupoleHandler::addK2(PacGenElement& genElement,
				      double l,
				      const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlK2  = attrs.getValue(m_chK2);
    if(xmlK2 == 0) return;

    char* chK2  = XMLString::transcode(xmlK2);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chK2);
    double k2 = muParser.Eval();

    XMLString::release(&chK2);

    if(k2 == 0.0) return;

    PacElemMultipole mult(2);
    mult.kl(2) = k2*l/2.0;
    genElement.add(mult);

    // std::cout << "UAL::ADXFElementHandler::addK2 " << k2 << std::endl;
}


