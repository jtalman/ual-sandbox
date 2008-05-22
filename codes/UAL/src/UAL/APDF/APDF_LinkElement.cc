// Library       : UAL
// File          : UAL/APDF/APDF_AlgorithmElement.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream>

#include "UAL/APF/PropagatorFactory.hh"
#include "UAL/APDF/APDF_LinkElement.hh"

UAL::APDF_LinkElement::APDF_LinkElement()
{

  m_type = UAL::APDF_LinkElement::EMPTY;

  m_rePattern = 0;
  m_pePattern = 0;
  
}

UAL::APDF_LinkElement::~APDF_LinkElement()
{
}

int UAL::APDF_LinkElement::getType() const
{
  return m_type;
}

void UAL::APDF_LinkElement::init(xmlNodePtr linkNode)
{

  // Algorithm

  xmlChar *algName =  xmlGetProp(linkNode, (const xmlChar *) "algorithm");
  if(algName == 0) {
    m_type = APDF_LinkElement::EMPTY;
    return;
  }

  UAL::PropagatorNode* prop = 
    UAL::PropagatorFactory::getInstance().createPropagator((const char*) algName);
  UAL::PropagatorNodePtr algPtr(prop);

  if(!algPtr.isValid()) {
    std::cout << "APBuilder error: algorithm " << (const char*) algName  
	      << " has not been found" << std::endl;
    m_type = APDF_LinkElement::EMPTY;
    xmlFree(algName);
    return;
  }

  m_classname = (const char*) algName;
  xmlFree(algName);
  m_algPtr = algPtr;  

  // Type and Element name pattern

  const char *error;
  int erroffset;

  // Sector
 
  xmlChar *sectorPattern =  xmlGetProp(linkNode, (const xmlChar *) "sector");

  if(sectorPattern != 0) {
    m_strPattern = (const char*) sectorPattern;
    xmlFree(sectorPattern);

    m_type = APDF_LinkElement::SECTOR;
    std::string::size_type ic = m_strPattern.find(",");
    std::string::size_type ie = m_strPattern.length();
    if(ic == std::string::npos){
      m_frontName = m_strPattern;
      m_backName  = m_strPattern;  
    } else {
      m_frontName = m_strPattern.substr(0, ic);
      m_backName  = m_strPattern.substr(ic + 1, ie - ic - 1); 
    } 
 
    return;
  }  

  // Element pattern

  xmlChar *elemPattern =  xmlGetProp(linkNode, (const xmlChar *) "elements");

  if(elemPattern != 0) {

    m_strPattern = (const char*) elemPattern;
    xmlFree(elemPattern);

    m_type = APDF_LinkElement::ELEMENT;
    m_rePattern = pcre_compile(m_strPattern.c_str(), 0, &error, &erroffset, NULL);
    m_pePattern = pcre_study(m_rePattern, 0, &error);
    return;
  }

  // Type pattern

  xmlChar *typePattern =  xmlGetProp(linkNode, (const xmlChar *) "types");

  if(typePattern != 0) {

    m_strPattern = (const char*) typePattern;
    xmlFree(typePattern);

    m_type = APDF_LinkElement::TYPE;    
    m_rePattern = pcre_compile(m_strPattern.c_str(), 0, &error, &erroffset, NULL);
    m_pePattern = pcre_study(m_rePattern, 0, &error);
  }

}
