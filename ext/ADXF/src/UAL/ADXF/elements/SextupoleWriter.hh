//# Library     : UAL
//# File        : UAL/ADXF/elements/SextupoleWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_SEXTUPOLE_WRITER_HH
#define UAL_ADXF_SEXTUPOLE_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFSextupoleWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFSextupoleWriter();

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
