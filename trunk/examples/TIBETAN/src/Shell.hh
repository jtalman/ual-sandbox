#ifndef UAL_APPS_GT_SHELL_HH
#define UAL_APPS_GT_SHELL_HH

#include <vector>

#include "UAL/APF/AcceleratorPropagator.hh"
#include "UAL/APF/PropagatorComponent.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacSmf.h"
#include "SMF/PacElemRfCavity.h"
#include "TIBETAN/Propagator/RFCavityTracker.hh"
#include "APPS/GT/JRampWindowController.hh"
#include "APPS/GT/JMountainRangeWindowController.hh"
#include "APPS/GT/WallCurrentMonitor.hh"

namespace GT {

  class Shell
  {
  public:

    /** Constructor */
    Shell();

    /** Destructor */
    ~Shell();

    /** Reads the original file */
    void readInputFile(const char* file);

    /** Reads the sxf file with accelerator lattice */
    void readSXFFile(const char* sxfFile, const char* accName,
		     const char* outDirName);

    /** Reads the adxf file with the accelerator propagator structure */
    void readAPDFFile(const char* adxfFile);

    /** Selects the rf cavity tracker */
    void selectRFCavity(const char* rfName);

    /** Make bunch distribution */
    void initBunch(double ctHalfWidth, double deHalfWidth);

    /** Tracking */
    void track();

    /** Opens application window */
    void openWindow();

    /** Closes application window */
    void closeWindow();

  protected:
    
    /** Initializes collections of the RF time-dependent parameters */
    // void initRFScenario();

    /** Updates the RF parameters */
    // void updateRF(double elapsedTime);

    /** Calculates gt */
    double getGT();

    /** Calculates alpha1 */
    double getAlpha1();

    /** Performs momentum scraping */
    int performMomentumScraping(PAC::Bunch& bunch);

    /** Shows profile */
    void showProfile(PAC::Bunch& bunch, double elapsedTime);

  protected:

    /** Bunch */
    PAC::Bunch m_bunch;

    /** Accelerator propagator built from the ADXF file */
    UAL::AcceleratorPropagator* m_ap;  

    /** Selected rf cavity tracker */
    TIBETAN::RFCavityTracker* m_rfTracker;

    /** Array of RF time-dependent parameters */
    std::vector<double> m_rfLagTimes;
    std::vector<double> m_rfLagValues;    
    
    /** Wall current monitor */
    GT::WallCurrentMonitor m_wcMonitor;

    /** Proxy of the mountain range plot */
    GT::JMountainRangeWindowController m_profileProxy; 

    /** Proxy of the ramp plot */
    GT::JRampWindowController m_rampProxy;  


  private:

    // 1st line
    int rauschen;            // 0 - without rf noise options
    int if_rausch;           // 0 - without rf noise detailed options

    // 2nd line
    int ifcoup;              // 1 - with collective effects; 0 - otherwise
    int ifhigh;              // 1 - with second to fourth order nonlinear momentum 
                             //     compaction factor; 0 with second order only
    int ifjmp;               // 1 - synchronous phase jumped transition crossing; 0 - otherwise
    int iffdk;               // 1 - with longitudinal feedback; 0 - otherwise

    // 3rd line
    int n;                   // number of macro particles for simulation
    double aconst;           // bunch are in [eV sec/A]; inactive for inmode = 7
    double scale;            // resolution scale for distribution; ~ 0.1
    int iran;                // random number seed for distribution
 
    // 4th line
    double z;                // electric charge state
    double a;                // atomic mass number
    double h;                // harmonic number
    int    nvmax;            // number of step points for RF voltage; < 20
    double phis;             // inactive (synchronous phase)
    double r0;               // C0/2/pi; C0 is circumference in [m]
    double prfstp;           // momentum step in [MeV] for digital rf frequency update;

    // 5th line
    std::vector<double> vr;  // cavity voltage in [V] at time tr[i]; i = 1, ..., nvmax

    // 6th line
    std::vector<double> tr;  // time in [sec]

    // 7th line
    double gami;             // initial gamma for tracking
    double gamf;             // final gamma for tracking
    int nout;                // print output every nout turns
    int ifpsw;               // 1 - print information on phase switching
    int ifbuck;              // 1 - print the RF bucket every output of bunch distribution
    int ibuc;                // inactive

    // 8th line
    int ikin;                // 0 - tracking from gami to gamf
                             // 1 - tracking from dt1 to gamf
                             // 2 - tracking from dt1 to dt2
    double dt0;              // time in [sec] before acceleration
    double pmin;             // momentum in [MeV/c] during acceleration ramp
    double pone;             // -"-
    double ptwo;             // -"-
    double pmax;             // -"-
    double detr;             // energy gain per turn in linear part of acceleration
                             // beta*de; dp in [MeV/c] momentum gain per turn

    // 9th line
    double dt1;              // starting time in [sec] for tracking (if ikin != 0)
    double dt2;              // final time in [sec] for tracking (if ikin != 0)

    // 10th line
    int inmode;              // initial distribution choice (1 - Gaussian, 2 - uniform,
                             // 3 - elliptical contour, 4 - first order parabola, 
                             // 5 - Lorentz, 6 - from input, 7 - DC beam
    int inmode2;             // (for inmode = 7 DC beam) 
                             // 1 - uniform in phase and momentum
                             // 2 - Gaussian in momentum, uniform in phase
    double ds0;              // (for inmode = 7 DC beam) center of momentum deviation
    double ph0;              // (for inmode = 7 DC beam) center of phase deviation
    double ds1;              // (for inmode = 7 DC beam) momentum spread

    // 11th line
    int ifcut;               // 1 - particle outside of RF bucket are thrown away if ifcoun = 1
                             // 2 - particles with dp > delap are thrown away if ifcoun = 1
    int ncut;                // cut away tail particles (ifcut = 1,2; ifcoun = 1) every ncut turns
                             // 0 - otherwise
    int ifcoun;              // 1 - perform momentum scraping
    double delap;            // momentum aperture for (ifcut = 2, ifcoun = 1)
    double tranap;           // transverse aperture in [m], used with dispersion ramp when ifhigh=1
    int ifprin;              // 1 - print detailed particle distribution (file9); 0 - otherwise

    // 12th line
    int ifpj;                // 1 - projection of particle distribution onto phase and momentum axis
    int npjscl;              // number of bins for projection
    int ndtgap;              // projection after every ndtgap turns
    int ifot3;               // 1 - bucket and bunch distribution are written on file 10 upon 
                             //     each output; 0 - otherwise
    int ifot4;               // 1 - initial bucket and bunch distribution are written on file 11; 
                             // 0 - otherwise
    // 13th line
    double gt0;              // central transition energy
    double gtsw;             // central gamma when the synchronous phase is switched
    double tswit;            // time in [sec] durin which the synchronous phase is switched (ifphis = 0)
    int ifphis;              // 1 - programmed phase jump near transition crossing
                             // if (ifphis = 1) phist, tst, vst, vst2: transition phase details

    // 14th line
    double dgt;              // alpha1 - second-order momentum compaction factor


  private:

    void init();


  };

};

#endif
