//# Library     : UAL
//# File        : UAL/ADXF/elements/RcollimatorWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_RCOLLIMATOR_WRITER_HH
#define UAL_ADXF_RCOLLIMATOR_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFRcollimatorWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFRcollimatorWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
