//# Library     : UAL
//# File        : UAL/ADXF/elements/MarkerWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_MARKER_WRITER_HH
#define UAL_ADXF_MARKER_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFMarkerWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFMarkerWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);
    
  };

}


#endif
