// Library       : UAL
// File          : UAL/APDF/APDF_CreateElement.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_APDF_CREATE_ELEMENT_HH
#define UAL_APDF_CREATE_ELEMENT_HH

#include <vector>

#include <pcre.h>
#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#include "UAL/APDF/APDF_LinkElement.hh"

namespace UAL {

  /** Container of the APDF "create" element. */

  class APDF_CreateElement {

  public:

    /** Constructor */
    APDF_CreateElement();

    /** Destructor */
    virtual ~APDF_CreateElement();

    /** Sets data */
    void init(xmlNodePtr node);

    /** Returns the  sector link specified by the front element name*/
    APDF_LinkElement& selectSectorLink(const std::string& elname);

    /** Returns the  element link specified by the element name*/
    APDF_LinkElement& selectElementLink(const std::string& elname);

    /** Returns the  element link specified by the element type*/
    APDF_LinkElement& selectTypeLink(const std::string& eltype);

  public:

    /** Collection of sector links */
    std::vector<APDF_LinkElement> m_sectorLinks;

    /** Collection of element links */
    std::vector<APDF_LinkElement> m_elementLinks;

    /** Collection of type links */
    std::vector<APDF_LinkElement> m_typeLinks;
    
  private:

    static APDF_LinkElement s_emptyLink;

    // for regular expression
    int m_overtor[30]; 

  };

}

#endif
