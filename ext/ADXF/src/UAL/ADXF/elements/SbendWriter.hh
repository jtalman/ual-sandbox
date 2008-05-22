//# Library     : UAL
//# File        : UAL/ADXF/elements/SbendWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_SBEND_WRITER_HH
#define UAL_ADXF_SBEND_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFSbendWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFSbendWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab);

  protected:

    void writeDesignBend(ostream& out, 
			 PacElemAttributes& atts,
			 const std::string& tab); 
    
  };

}


#endif
