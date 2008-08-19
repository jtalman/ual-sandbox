
#include <iostream>

#include "xercesc/sax2/Attributes.hpp"

#include "SMF/PacLine.h"
#include "SMF/PacLattice.h"
#include "SMF/PacGenElements.h"
#include "SMF/PacSmf.h"
#include "SMF/PacElemLength.h"

#include "UAL/ADXF/ConstantManager.hh"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/SectorHandler.hh"

XERCES_CPP_NAMESPACE_USE

double UAL::ADXFSectorHandler::s_diff = 1.0e-5;

UAL::ADXFSectorHandler::ADXFSectorHandler()
{

  m_pLattice = 0;

  m_chName  = new XMLCh[100];
  XMLString::transcode("name", m_chName, 99);
  m_chLine  = new XMLCh[100];
  XMLString::transcode("line", m_chLine, 99);

  m_chSector = new XMLCh[100];
  XMLString::transcode("sector", m_chSector, 99);
  m_chFrame  = new XMLCh[100];
  XMLString::transcode("frame", m_chFrame, 99);

  m_chAt  = new XMLCh[100];
  XMLString::transcode("at", m_chAt, 99);
  m_chRef  = new XMLCh[100];
  XMLString::transcode("ref", m_chRef, 99);

}

UAL::ADXFSectorHandler::~ADXFSectorHandler()
{
  XMLString::release(&m_chName);
  XMLString::release(&m_chLine);

  XMLString::release(&m_chSector);
  XMLString::release(&m_chFrame);

  XMLString::release(&m_chAt);
  XMLString::release(&m_chRef);
}

void UAL::ADXFSectorHandler::startElement(const   XMLCh* const    uri,
					  const   XMLCh* const  localname,
					  const   XMLCh* const    qname,
					  const   xercesc::Attributes&     attrs)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSectorHandler::startElement: saw element: "<< message << endl;
    XMLString::release(&message);

    if(XMLString::compareString(localname, m_chFrame) == 0){
      addLatticeElement(attrs);
      return;
    }

    const XMLCh* xmlName  = attrs.getValue(m_chName);
    char* name    = XMLString::transcode(xmlName);

    // std::cout << "line name=" << name << std::endl;

    PacLine line(name);
    PacLattice lattice(name);

    bool isLine = checkLine(attrs);
    if(isLine) {
      addLine(line, lattice, attrs);
    }

    PacLattices* lattices = PacLattices::instance();

    PacLattices::iterator it = lattices->find(name);
    if(it == lattices->end()){
      std::cout << "lattice " << name << " is not created " << std::endl;
    }

    m_pLattice = &(*it);
    m_ElementList.clear();

    m_at = 0;
    m_iDriftCounter = 0;

    XMLString::release(&name);

}

void UAL::ADXFSectorHandler::endElement(const   XMLCh* const    uri,
					const   XMLCh* const    localname,
					const   XMLCh* const    qname)
{
    char* message = XMLString::transcode(localname);
    // cout << "ADXFSectorHandler::endElement: saw element: "<< message << endl;
    XMLString::release(&message);

    if(XMLString::compareString(localname, m_chSector) == 0){

      if(m_pLattice != 0 && m_ElementList.size() > 0 ) m_pLattice->set(m_ElementList);

      m_ElementList.clear();
      m_pLattice = 0;
    }

    // UAL::ADXFReader::getInstance()->getSAX2Reader()->setContentHandler(p_parentHandler);
    // UAL::ADXFReader::getInstance()->getSAX2Reader()->setErrorHandler(p_parentHandler);
}

void UAL::ADXFSectorHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "ADXFSectorHandler: Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

void UAL::ADXFSectorHandler::addLatticeElement(const xercesc::Attributes& attrs)
{
    const XMLCh* xmlRef  = attrs.getValue(m_chRef);
    char* ref    = XMLString::transcode(xmlRef);

    const XMLCh* xmlAt  = attrs.getValue(m_chAt);
    char* chAt    = XMLString::transcode(xmlAt);

    mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

    muParser.SetExpr(chAt);
    double at = muParser.Eval();

    if(at  > m_at + s_diff) {

      PacLattElement drift;
      
      double driftL = at - m_at;
      m_DriftLength.l(driftL); 

      std::cout << "addLatticeElement " 
		<< "ref=" << ref 
		<< ", at =" << at 
		<< ", m_at = " << m_at << std::endl;

      char sCounter[5];
      sprintf(sCounter, "%d", m_iDriftCounter++);
      drift.name("_" + m_pLattice->name() + "_" + sCounter);

      drift.set(m_DriftLength);

      m_ElementList.push_back(drift); 

      m_at = at;
      
    } else if( (at + s_diff) < m_at) {
      std::cout << "problem at(" << at << ") < (" << m_at << ") " << std::endl;
    }

    // std::cout << "addLatticeElement " << "ref=" << ref << ", at =" << chAt << std::endl;

    PacGenElements* genElements = PacGenElements::instance();
    PacGenElements::iterator it = genElements->find(ref);
    if(it == genElements->end()){
      std::cout << "there is no generic element for reference " << ref << std::endl;
    }
     
 
    PacGenElement& genElement = (*it);
    PacLattElement lattElement(genElement);
  
    m_ElementList.push_back(lattElement);
    m_at = at + lattElement.getLength();
 
    XMLString::release(&ref);
    XMLString::release(&chAt);
}

bool UAL::ADXFSectorHandler::checkLine(const xercesc::Attributes& attrs)
{
   const XMLCh* xmlLine = attrs.getValue(m_chLine);
   if(xmlLine == 0) return false;
   return true;
}

void UAL::ADXFSectorHandler::addLine(PacLine& line, 
				     PacLattice& lattice,
				     const xercesc::Attributes& attrs)
{

    const XMLCh* xmlLine = attrs.getValue(m_chLine);
    char* chLine = XMLString::transcode(xmlLine);

    std::string strLine(chLine);

    std::vector<std::string> tokens;
    tokenize(strLine, tokens); 

    PacGenElements* genElements = PacGenElements::instance();
    PacLines* lines = PacLines::instance();

    for(unsigned int it=0; it < tokens.size(); it++){

      // std::cout << it << " " << tokens[it] << std::endl;

      PacGenElements::iterator ige = genElements->find(tokens[it]);
      if(ige != genElements->end()){
	// std::cout << "add gen element" << std::endl;
	line.add(*ige);	
	continue;
      } 
      
      PacLines::iterator il = lines->find(tokens[it]);
      if(il != lines->end()) {
	// std::cout << "add line" << std::endl;	
	line.add(*il);
	continue;
      }
	
      std::cout << "there is no genElement or line " << tokens[it] << std::endl;
      exit(0);

    }

    lattice.set(line);
    XMLString::release(&chLine);
}


void UAL::ADXFSectorHandler::tokenize(const std::string& str,
				      std::vector<std::string>& tokens,
				      const std::string& delimiters)
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}


