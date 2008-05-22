//# Library     : UAL
//# File        : UAL/ADXF/elements/HMonitorWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_HMONITOR_WRITER_HH
#define UAL_ADXF_HMONITOR_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFHmonitorWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFHmonitorWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
