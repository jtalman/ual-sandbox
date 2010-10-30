// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/ElectricData.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_ELECTRIC_DATA_HH
#define UAL_ETEAPOT_ELECTRIC_DATA_HH

#include "SMF/PacLattElement.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacElemMultipole.h"
#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "SMF/PacElemAperture.h"

namespace ETEAPOT {

  /** Collection of attributes used by the ETEAPOT electric propagator */

  class ElectricData {

  public:

    /** Constructor */
    ElectricData();

    /** Copy constructor */
    ElectricData(const ElectricData& edata);

    /** Destructor */
    ~ElectricData();

    /** Copy operator */
    const ElectricData& operator=(const ElectricData& edata);

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  public:

    /** Entry multipole */
    PacElemMultipole *m_entryMlt;
    
    /** Exit multipole */
    PacElemMultipole *m_exitMlt;

    /** Multipole attributes */
    PacElemMultipole* m_mlt;

    /** Offset */
    PacElemOffset* m_offset;
    
    /** Rotation */
    PacElemRotation* m_rotation;

    // Aperture 
    // PacElemAperture* m_aperture; 

  private:

    void initialize();
    void copy(const ElectricData& edata);
 
  };

}

#endif
