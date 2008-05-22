//# Library     : UAL
//# File        : UAL/ADXF/Reader.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_READER_HH
#define UAL_ADXF_READER_HH

#include <xercesc/sax2/SAX2XMLReader.hpp>

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/DocumentHandler.hh"

namespace UAL {

  /**
   * The Reader class initializes the SMF containers from the ADXF file.
   */ 

  class ADXFReader
  {
  public:

    /** Returns a singleton */
    static ADXFReader* getInstance();

    /** Destructor */
    ~ADXFReader();

    /** Writes data. */
    void read(const char* outFile);

    xercesc::SAX2XMLReader* getSAX2Reader() { return p_reader; }

  protected:

    std::string m_tab;
    xercesc::SAX2XMLReader* p_reader;

  private:

    /** Constructor. */
    ADXFReader();

    static ADXFReader* s_theInstance;

    ADXFDocumentHandler m_docHandler;

  };

}

#endif
