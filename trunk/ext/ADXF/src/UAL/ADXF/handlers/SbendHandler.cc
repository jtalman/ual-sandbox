
#include <iostream>

#include "SMF/PacElemBend.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/SbendHandler.hh"
#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFSbendHandler::ADXFSbendHandler()
{
  m_chAngle  = new XMLCh[100];
  XMLString::transcode("angle", m_chAngle, 99);
}

UAL::ADXFSbendHandler::~ADXFSbendHandler()
{
  XMLString::release(&m_chAngle);
}

void UAL::ADXFSbendHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSbendHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    PacSbend genElement(name);

    XMLString::release(&name);

    addLength(genElement, attrs);
    addAngle(genElement, attrs);
}

void UAL::ADXFSbendHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSbendHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFSbendHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFSbendHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFSbendHandler::addAngle(PacGenElement& genElement, 
				     const   xercesc::Attributes& attrs)
{
    const XMLCh* xmlAngle  = attrs.getValue(m_chAngle);
    if(xmlAngle == 0) return;

    char* chAngle  = XMLString::transcode(xmlAngle);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chAngle);
    double angle = muParser.Eval();

    XMLString::release(&chAngle);

    if(angle == 0.0) return;

    PacElemBend bend;
    bend.angle() = angle;
    genElement.add(bend);

    // std::cout << "UAL::ADXFElementHandler::addAngle " << angle << std::endl;
}


