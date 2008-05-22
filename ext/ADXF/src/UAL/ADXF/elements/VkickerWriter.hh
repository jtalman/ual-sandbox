//# Library     : UAL
//# File        : UAL/ADXF/elements/VkickerWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_VKICKER_WRITER_HH
#define UAL_ADXF_VKICKER_WRITER_HH

#include "UAL/ADXF/ElementWriter.hh"

namespace UAL 
{

  class ADXFVkickerWriter : public ADXFElementWriter
  {
  public:

    /** Constructor */
    ADXFVkickerWriter();

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
