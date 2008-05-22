//# Library     : UAL
//# File        : UAL/ADXF/ElementWriter.hh
//# Copyright   : see Copyrigh file

#ifndef UAL_ADXF_ELEMENT_WRITER_HH
#define UAL_ADXF_ELEMENT_WRITER_HH

#include "UAL/ADXF/Def.hh"

namespace UAL {

  /** 
   * Basic class of various element writers.
   */

  class ADXFElementWriter 
  {
  public:

    /** Constructor */
    ADXFElementWriter();

    /** Destructor */
    virtual ~ADXFElementWriter();

    /** Writes design elements  into an output stream. */
    virtual void writeDesignElement(ostream& out, PacGenElement& element,
      const std::string& tab) {}

  protected:

    double getLength(PacGenElement& elem);

  };

}

#endif
