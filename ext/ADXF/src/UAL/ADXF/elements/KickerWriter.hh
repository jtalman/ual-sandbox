//# Library     : UAL
//# File        : UAL/ADXF/elements/KickerWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_KICKER_WRITER_HH
#define UAL_ADXF_KICKER_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFKickerWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFKickerWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);

  protected:

    void writeDesignMultipole(ostream& out, 
			      PacElemAttributes& atts,
			      const std::string& tab);    
    
  };

}


#endif
