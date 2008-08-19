//# Library     : UAL
//# File        : UAL/ADXF/SectorHandler.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_SECTOR_HANDLER_HH
#define UAL_ADXF_SECTOR_HANDLER_HH

#include <vector>

#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLChar.hpp>

#include "SMF/PacElemLength.h"

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/BasicHandler.hh"

namespace UAL {

  /**
   * The handler of the accelerator sector
   */ 

  class ADXFSectorHandler : public ADXFBasicHandler
  {
  public:

    /** Constructor */
    ADXFSectorHandler();

    /** Destructor */
    ~ADXFSectorHandler();

    void startElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname,
        const   xercesc::Attributes&     attrs
    );

    void endElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname
    );

    void fatalError(const xercesc::SAXParseException&);

  protected:

    void addLatticeElement(const xercesc::Attributes& attrs);
    bool checkLine(const   xercesc::Attributes&     attrs);
    void addLine(PacLine& line, PacLattice& lattice, 
		 const   xercesc::Attributes&     attrs);
    void tokenize(const std::string& str,
		  std::vector<std::string>& tokens,
		  const std::string& delimiters = " ");

  protected:

    // ADXFFrameHandler m_frameHandler;   

    XMLCh* m_chName;
    XMLCh* m_chLine;

    XMLCh* m_chSector;
    XMLCh* m_chFrame;

    XMLCh* m_chAt;
    XMLCh* m_chRef;

  protected:

    static double s_diff;

    /** Pointer to the SMF lattice. */
    PacLattice* m_pLattice;

    /** List of elements. */
    PacList<PacLattElement> m_ElementList;

    /** Drift counter. */
    int m_iDriftCounter;

    /** Current position */
    double m_at;

    /** Drift bucket. */
    PacElemLength m_DriftLength; 
  };

}

#endif
