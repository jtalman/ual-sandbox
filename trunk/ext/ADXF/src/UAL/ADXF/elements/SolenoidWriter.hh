//# Library     : UAL
//# File        : UAL/ADXF/elements/SolenoidWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_SOLENOID_WRITER_HH
#define UAL_ADXF_SOLENOID_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFSolenoidWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFSolenoidWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, 
				    PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
