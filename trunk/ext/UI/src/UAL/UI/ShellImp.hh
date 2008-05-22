#ifndef UAL_SHELL_IMP_HH
#define UAL_SHELL_IMP_HH

#include "ZLIB/Tps/Space.hh"
#include "Main/Teapot.h"

namespace UAL {

  /** Container with the UAL old classes used in the Shell algorithms */

  class ShellImp {

  public:

    /** Returns singleton */
    static ShellImp& getInstance();

  public:

    /** Pointer to ZLIB space of Tps objects */
    ZLIB::Space* m_space;

    /** SMF Lattice */
    PacLattice m_lattice;

    /** TEAPOT Shell */
    Teapot m_teapot;

  protected:

  private:

    /** Constructor */
    ShellImp();

  private:

    static ShellImp* s_theInstance;

  };

}

#endif
