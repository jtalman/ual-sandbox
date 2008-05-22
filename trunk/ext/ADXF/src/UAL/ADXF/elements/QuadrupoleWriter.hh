//# Library     : UAL
//# File        : UAL/ADXF/elements/QuadrupoleWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_QUADRUPOLE_WRITER_HH
#define UAL_ADXF_QUADRUPOLE_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFQuadrupoleWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFQuadrupoleWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);

  protected:

    void writeDesignMultipole(ostream& out, 
			      const std::string& name, 
			      PacElemAttributes& atts,
			      const std::string& tab);    
    
  };

}


#endif
