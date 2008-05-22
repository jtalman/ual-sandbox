//# Library     : UAL
//# File        : UAL/ADXF/SectorsWriter.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_SECTORS_WRITER_HH
#define UAL_ADXF_SECTORS_WRITER_HH

#include <map>

#include "UAL/ADXF/Def.hh"
#include "UAL/ADXF/SectorWriter.hh"

namespace UAL {

  /** 
   * Writer of ADXF sectors.
   */

  class ADXFSectorsWriter 
  {
  public:

    /** Constructor.*/
    ADXFSectorsWriter();   

    /** Destructor */
    ~ADXFSectorsWriter();     

    /** Writes design elements  into an output stream. */
    void writeSectors(ostream& out, const std::string& tab);

  protected:

    ADXFSectorWriter m_sectorWriter;

  };

}

#endif
