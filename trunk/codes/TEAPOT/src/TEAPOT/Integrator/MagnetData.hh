// Library       : TEAPOT
// File          : TEAPOT/Integrator/MagnetData.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_MAGNET_DATA_HH
#define UAL_TEAPOT_MAGNET_DATA_HH

#include "SMF/PacLattElement.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacElemMultipole.h"
#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "SMF/PacElemAperture.h"

namespace TEAPOT {

  /** Collection of attributes used by the TEAPOT magnet propagator */

  class MagnetData {

  public:

    /** Constructor */
    MagnetData();

    /** Copy constructor */
    MagnetData(const MagnetData& data);

    /** Destructor */
    ~MagnetData();

    /** Copy operator */
    const MagnetData& operator=(const MagnetData& data);

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
    void copy(const MagnetData& mdata);
 
  };

}

#endif
