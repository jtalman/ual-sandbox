//# Library     : UAL
//# File        : UAL/ADXF/ElementsWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_ELEMENTS_WRITER_HH
#define UAL_ADXF_ELEMENTS_WRITER_HH

#include <map>

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/ElementWriter.hh"

namespace UAL {

  /** 
   * Writer of ADXF design and real elements.
   */

  class ADXFElementsWriter 
  {
  public:

    /** Constructor.*/
    ADXFElementsWriter();   

    /** Destructor */
    ~ADXFElementsWriter();     

    /** Writes design elements  into an output stream. */
    void writeDesignElements(ostream& out, const std::string& tab);

  protected:

    std::map<std::string, ADXFElementWriter*> m_elemWriters;

  };

}

#endif
