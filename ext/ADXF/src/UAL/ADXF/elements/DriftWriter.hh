//# Library     : UAL
//# File        : UAL/ADXF/elements/DriftWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_DRIFT_WRITER_HH
#define UAL_ADXF_DRIFT_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFDriftWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFDriftWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
