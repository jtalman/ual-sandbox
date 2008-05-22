//# Library     : UAL
//# File        : UAL/ADXF/elements/MonitorWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_MONITOR_WRITER_HH
#define UAL_ADXF_MONITOR_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFMonitorWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFMonitorWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
