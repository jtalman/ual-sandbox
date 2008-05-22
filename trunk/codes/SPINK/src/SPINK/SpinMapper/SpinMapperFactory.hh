// Library       : SPINK
// File          : SPINK/SpinMatrix/SpinMapperFactory.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_SPIN_MAPPER_FACTORY_HH
#define UAL_SPINK_SPIN_MAPPER_FACTORY_HH

#include "SPINK/SpinMapper/DriftSpinMapper.hh"
// #include "SPINK/SpinMapper/MagnetSpinMapper.hh"

namespace SPINK {

  /** Factory of the Spin Mappers */

  class SpinMapperFactory {

  public:

    /** Returns the spin mapper specified by type */
    // static SpinMapper* createMapper(const std::string& type);

    /** Returns the default spin mapper */
    static SpinMapper* createDefaultMapper();

    /** Returns the drift spin mapper */
    static DriftSpinMapper* createDriftMapper();

    /** Returns the magnet spin mapper */
    // static MagnetSpinMapper* createMagnetMapper();

  };

  class SpinMapperRegister 
  {
    public:

    SpinMapperRegister(); 
  };

}

#endif
