// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltData.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_MLT_DATA_HH
#define ETEAPOT_MLT_DATA_HH

#include "SMF/PacLattElement.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacElemMultipole.h"
#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "SMF/PacElemAperture.h"

namespace ETEAPOT {

  /** Collection of attributes used by the ETEAPOT mlt propagator */

  class MltData {

  public:

    /** Constructor */
    MltData();

    /** Copy constructor */
    MltData(const MltData& edata);

    /** Destructor */
    ~MltData();

    /** Copy operator */
    const MltData& operator=(const MltData& edata);

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
    void copy(const MltData& edata);
 
  };

}

#endif
