//# Library     : UAL
//# File        : UAL/ADXF/elements/MultipoleWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_MULTIPOLE_WRITER_HH
#define UAL_ADXF_MULTIPOLE_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFMultipoleWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFMultipoleWriter();

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
