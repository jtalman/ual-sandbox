//# Library     : UAL
//# File        : UAL/ADXF/SectorWriter.hh
//# Copyright   : see Copyrigh file

#ifndef UAL_ADXF_SECTOR_WRITER_HH
#define UAL_ADXF_SECTOR_WRITER_HH

#include "UAL/ADXF/Def.hh"

namespace UAL {

  /** 
   * Sector writer.
   */

  class ADXFSectorWriter 
  {
  public:

    /** Constructor */
    ADXFSectorWriter();

    /** Destructor */
    virtual ~ADXFSectorWriter();

    /** Writes SMF lattices  into an output stream. */
    virtual void writeSector(ostream& out, 
			     PacLattice& lattice,
			     const std::string& tab);
  protected:



  };

}

#endif
