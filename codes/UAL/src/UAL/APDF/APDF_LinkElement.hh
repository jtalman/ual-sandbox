// Library       : UAL
// File          : UAL/APDF/APDF_LinkElement.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_APDF_LINK_ELEMENT_HH
#define UAL_APDF_LINK_ELEMENT_HH

#include <pcre.h>
#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#include "UAL/APF/PropagatorNode.hh"

namespace UAL {

  /** Container of the APDF "link" element */

  class APDF_LinkElement {

  public:

    enum LINK_TYPES {
      EMPTY = 0,
      SECTOR,
      ELEMENT,
      TYPE
    };

    /** Constructor */
    APDF_LinkElement();

    /** Destructor */
    virtual ~APDF_LinkElement();

    /** Sets data */
    void init(xmlNodePtr linkNode);

    /** Returns a link type */
    int getType() const;

  public:

    /** Link type */
    int m_type;

    /** Smart pointer of the propagator node */
    PropagatorNodePtr m_algPtr;

    /** classname */
    std::string m_classname;

    /** front element name */
    std::string m_frontName;

    /** back element name */
    std::string m_backName;

    /** pattern */
    std::string m_strPattern;

    /** Regular expression of the pattern*/
    pcre* m_rePattern;

    /** Extra stuff*/
    pcre_extra *m_pePattern;
    
  };

}

#endif
