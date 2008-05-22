//# Library     : UAL
//# File        : UAL/ADXF/elements/InstrumentWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_INSTRUMENT_WRITER_HH
#define UAL_ADXF_INSTRUMENT_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFInstrumentWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFInstrumentWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
