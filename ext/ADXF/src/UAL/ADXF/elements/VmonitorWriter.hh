//# Library     : UAL
//# File        : UAL/ADXF/elements/VmonitorWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_VMONITOR_WRITER_HH
#define UAL_ADXF_VMONITOR_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFVmonitorWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFVmonitorWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
