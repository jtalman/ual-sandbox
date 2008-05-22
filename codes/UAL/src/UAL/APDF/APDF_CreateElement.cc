// Library       : UAL
// File          : UAL/APDF/APDF_CreateElement.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream>

#include "UAL/APF/PropagatorFactory.hh"
#include "UAL/APDF/APDF_CreateElement.hh"

UAL::APDF_LinkElement UAL::APDF_CreateElement::s_emptyLink;

UAL::APDF_CreateElement::APDF_CreateElement()
{
}

UAL::APDF_CreateElement::~APDF_CreateElement()
{
}

void UAL::APDF_CreateElement::init(xmlNodePtr createElement)
{

  xmlNodePtr cur = createElement->xmlChildrenNode;

  while(cur != 0){
    if ((!xmlStrcmp(cur->name, (const xmlChar *) "link"))){

      UAL::APDF_LinkElement linkElement;
      linkElement.init(cur);

      switch(linkElement.getType()){

      case UAL::APDF_LinkElement::SECTOR:
	m_sectorLinks.push_back(linkElement);
	break;
      case UAL::APDF_LinkElement::TYPE:
	m_typeLinks.push_back(linkElement);
	break;
      case UAL::APDF_LinkElement::ELEMENT:
	m_elementLinks.push_back(linkElement);
	break;
      default:
	break;
      }
    }
    cur = cur->next;
  }

}


UAL::APDF_LinkElement& UAL::APDF_CreateElement::selectSectorLink(const std::string& elname)
{

  for(unsigned int ia = 0; ia < m_sectorLinks.size(); ia++){
    if(!m_sectorLinks[ia].m_frontName.compare(elname)){
      return m_sectorLinks[ia];
    }
  }

  return s_emptyLink;
}

UAL::APDF_LinkElement& UAL::APDF_CreateElement::selectElementLink(const std::string& elname)
{

  for(unsigned int ia = 0; ia < m_elementLinks.size(); ia++){
    if(m_elementLinks[ia].m_rePattern == 0) continue;
    int rc = pcre_exec(m_elementLinks[ia].m_rePattern, 
		       m_elementLinks[ia].m_pePattern, 
		       elname.c_str(), elname.size(), 0, 0, m_overtor, 30);
    if(rc > 0){
      return m_elementLinks[ia];
    }
  }

  return s_emptyLink;
}

UAL::APDF_LinkElement& UAL::APDF_CreateElement::selectTypeLink(const std::string& eltype)
{

  for(unsigned int ia = 0; ia < m_typeLinks.size(); ia++){
    if(m_typeLinks[ia].m_rePattern == 0) continue;
    int rc = pcre_exec(m_typeLinks[ia].m_rePattern, 
		       m_typeLinks[ia].m_pePattern, 
		       eltype.c_str(), eltype.size(), 0, 0, m_overtor, 30);
    if(rc > 0){
      return m_typeLinks[ia];
    }
  }

  return s_emptyLink;
}

