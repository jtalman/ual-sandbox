
#include <iostream>

#include <xercesc/util/XMLStringTokenizer.hpp>

#include "SMF/PacElemMultipole.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/handlers/MfieldHandler.hh"

#include "UAL/ADXF/ConstantManager.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFMfieldHandler::ADXFMfieldHandler()
{
  m_chML  = new XMLCh[100];
  XMLString::transcode("ml", m_chML, 99);
  m_chA  = new XMLCh[100];
  XMLString::transcode("a", m_chA, 99);
  m_chB  = new XMLCh[100];
  XMLString::transcode("b", m_chB, 99);
}

UAL::ADXFMfieldHandler::~ADXFMfieldHandler()
{
  XMLString::release(&m_chML);
  XMLString::release(&m_chA);
  XMLString::release(&m_chB);
}

void UAL::ADXFMfieldHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const    localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    addField(attrs);
}

void UAL::ADXFMfieldHandler::endElement(const   XMLCh* const    uri,
					const   XMLCh* const    localname,
					const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSextupoleHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);


    UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFMfieldHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFMfieldHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFMfieldHandler::addField(const   xercesc::Attributes& attrs)
{
  // std::cout << "add field " << std::endl;
  // std::cout << "gen element name = " << p_genElement->name() << std::endl;

  mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

  std::vector<double> as;
  std::vector<double> bs;


  const XMLCh* xmlA  = attrs.getValue(m_chA);
  if(xmlA != 0) {
    XMLStringTokenizer tokensA(xmlA);
    int tokens = tokensA.countTokens();
    as.resize(tokens);
    int counter = 0;
    while (tokensA.hasMoreTokens()){
      const XMLCh* xmlAi = tokensA.nextToken();
      char* chAi  = XMLString::transcode(xmlAi);
      muParser.SetExpr(chAi);
      as[counter] = muParser.Eval();
      XMLString::release(&chAi);
      counter++;
    }
  }

  const XMLCh* xmlB  = attrs.getValue(m_chB);
  if(xmlB != 0) {
    XMLStringTokenizer tokensB(xmlB);
    int tokens = tokensB.countTokens();
    bs.resize(tokens);
    int counter = 0;
    while (tokensB.hasMoreTokens()){
      const XMLCh* xmlBi = tokensB.nextToken();
      char* chBi  = XMLString::transcode(xmlBi);
      muParser.SetExpr(chBi);
      bs[counter] = muParser.Eval();
      XMLString::release(&chBi);
      counter++;
    }
  }

  unsigned int size = bs.size();
  if(as.size() > size) size = as.size();

  if(size < 1) return;

  PacElemMultipole mult(size-1);

  for(unsigned int i=0; i < as.size(); i++){
    mult.ktl(i) = as[i];
  }

  for(unsigned int i=0; i < bs.size(); i++){
    mult.kl(i) = bs[i];
  }

  p_genElement->add(mult);

}


