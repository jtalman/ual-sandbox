#ifndef UAL_BTF_SHELL_HH
#define UAL_BTF_SHELL_HH

#include "AIM/BTF/BTFBpm.hh"
#include "AIM/BTF/BTFKicker.hh"

#include "RootShell.hh"

class TNtupleD;
class TGraph;
class TH2D;

  class BTFShell : public UAL::RootShell
  {

  public:

    /** Constructor */
    BTFShell();

    /** Destructor */
    ~BTFShell();

    /** Returns the accelerator length */
    double getLength();

    /** Returns a shell tarcker */
    UAL::AcceleratorPropagator* getTracker();

    /** Generates a particle bunch */
    void generateBunch(PAC::Bunch& bunch, double ctHalfWidth, double deHalfWidth, int iran);

    /** Returns a line density */
    void getLineDensity(TH2D& densityTH2D);

    /** Returns a horizontal  dipole driving term  */
    void getHDipoleTerm(TH2D& xdtermTH2D);

    /** Returns a x spectrum  */
    void getHSpectrum(TH2D& xspectrumTH2D, double revFreq, int resNumber);

    /** Calculates the slip factor */
    // double getSlipFactor();

    ClassDef(BTFShell, 1);

  };


#endif
