#ifndef UAL_SHELL_HH
#define UAL_SHELL_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/AcceleratorPropagator.hh"
#include "ZLIB/Tps/Space.hh"
#include "Main/Teapot.h"

#include "Arguments.hh"

namespace UAL {

  class Shell {

  public:

    /** Constructor */
    Shell();

    /**  Define the space of Taylor maps. */
    bool setMapAttributes(const Arguments& args);

    /**  Define the beam attributes. */
    bool setBeamAttributes(const Arguments& args);

    /**  Returns a container with  the beam attributes. */
    PAC::BeamAttributes& getBeamAttributes();

    /** Sets bunch distribution */
    bool setBunch(const Arguments& args);

    /** Reads SXF file with the lattice description */
    bool readSXF(const Arguments& args);

    /** Selects lattice */
    bool use(const Arguments& args);

    /** Make linear analysis */
    bool analysis(const Arguments& args);

    /** Reads APDF file with the propagator description  */
    bool readAPDF(const Arguments& args);

    /** Runs a bunch of particles */
    bool run(const Arguments& args);

    AcceleratorPropagator* getAcceleratorPropagator() { return m_ap; }


  private:

    ZLIB::Space* m_space;

    std::string m_accName;

    PAC::BeamAttributes m_ba;

    PacLattice m_lattice;

    Teapot m_teapot;

    UAL::AcceleratorPropagator* m_ap;

  };

};

#endif
