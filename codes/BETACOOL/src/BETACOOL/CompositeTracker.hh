#ifndef UAL_BETACOOL_COMPOSITE_TRACKER
#define UAL_BETACOOL_COMPOSITE_TRACKER

#include "PAC/Beam/Bunch.hh"
#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "xdynamic.h"

namespace BETACOOL
{
  /** Adapter to a collection of varios effects, such as electron cooling, ibs, etc. */

  class CompositeTracker
  {
  public:

    /** Constructor */
    CompositeTracker();

    /** Registers IBS effect */
    static void registerEffect(const char* effect, const char* ename);

   /** Sets time step */
   void setTimeStep(double timeStep);

    /** Propagates a bunch of particles */
    void propagate(PAC::Bunch& bunch);

    /** Set lattices */
    void setLattice(PacTwissData& twiss);

    /** Copies Betacool bunch into UAL container */
   void writeBunch(PAC::Bunch& bunch);

    /** Copies UAL bunch into Betacool container */
   void readBunch(PAC::Bunch& bunch);

  private:

   //xLattice lattice;

    // calculates histograms for loss calculation
    void calculateHistogram(xLattice& Lattice);

    // makes transverse rotation from lattice1 to lattice2
    void transRotate(xLattice& lattice1, xLattice& lattice2);

    // makes longitudinal rotation
    void longRotate();

    // adds kick of the ith effect
    void addKick(int i);

  };

};

#endif
