//# Library       : ACCSIM
//# File          : ACCSIM/Bunch/BunchAnalyzer.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_BUNCH_ANALYZER_HH
#define UAL_ACCSIM_BUNCH_ANALYZER_HH

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTwissData.h"
#include "ACCSIM/Base/Def.hh"

namespace ACCSIM {

  /** A collection of algorithms for
    calculating bunch integral parameters (rms, Twiss, etc.)
    based on the particle distribution.
  */ 

  class BunchAnalyzer
  {
  public:

    /** Constructor */
    BunchAnalyzer();

    /** Destructor */
    virtual ~BunchAnalyzer();

    /** Calculates the beam emittance */
    void getRMS(const PAC::Bunch& bunch, 
		PAC::Position& orbit,
		PacTwissData& twiss,
		PAC::Position& rms);

  private:
  
  };
}

#endif
