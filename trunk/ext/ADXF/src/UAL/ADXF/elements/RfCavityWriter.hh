//# Library     : UAL
//# File        : UAL/ADXF/elements/RfCavityWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_RFCAVITY_WRITER_HH
#define UAL_ADXF_RFCAVITY_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFRfCavityWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFRfCavityWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, 
				    PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
