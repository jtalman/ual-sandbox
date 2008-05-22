/* ****************************************************************************
   *                                                                          *
   *   This C++ code is an example of how to develop a ROOT-based shell to    *
   *   UAL simulation environment class can be                                *
   *                                                                          *
   *   To use the IO capabilities of ROOT, a class must:                      *
   *   1) Ultimately inherit from TObject.                                    *
   *   2) Use the macro ClassDef(classname,version) in the header file        *
   *   3) Use the macro ClassImp(classname) in the .cc file                   *
   *   4) the makefile must generate the dictionary/streamer file             *
   *   5) The class dictionary file must be linked to the class file          *
   *                                                                          *
   *   The ClassDef and ClassImp macros define other class members that       *
   *   are needed to use ROOT IO and RTTI facilities.  Classes that do        *
   *   no use these facilities do not need these modifications                *
   *                                                                          *
   *                                                                          *
   *   Author: Ray Fliller III and Nikolay Malitsky                           *
   *                                                                          *
   *                                                                          *
   **************************************************************************** */

#ifndef UAL_ROOT_SHELL_HH
#define UAL_ROOT_SHELL_HH

#include "TObject.h"
#include "PAC/Beam/Bunch.hh"
#include "PAC/Beam/BeamAttributes.hh"
#include "PAC/Beam/Position.hh"
#include "UAL/APF/AcceleratorPropagator.hh"
#include "UAL/SMF/AcceleratorNode.hh"
#include "ZLIB/Tps/Space.hh"
#include "Optics/PacTwissData.h"

#ifndef __CINT__
// hide these from CINT, CINT only sees the public parts.
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#endif


class TNtupleD;

namespace UAL {

  class RootShell : public TObject
  {

  public:

    /** Constructor */
    RootShell();

    /** Destructor */
    virtual ~RootShell();

    /** Reads the SXF file and builds the lattice */
    void readSXF(const char* inFile, const char* outDir);

    /** Defines the lattice */
    void use(const char* latticeName);

    /** Defines beam attributes */
    void setBeamAttributes(const PAC::BeamAttributes& ba);
    void setBeamAttributes(const double energy, const double mass, const double charge);

    PAC::BeamAttributes getBeamAttributes(){ return m_ba;}

    /** Generates a particle bunch matching Twiss parameters at element at*/
    void generateBunch(PAC::Bunch& bunch, int at=0);

    /** Reads the ADXF file and builds the tracker */
    void readAPDF(const char* inFile);

    /** Tracks particles */
    void track(PAC::Bunch &bunch, int start=0, int end=-1);
    void multitrack(PAC::Bunch &bunch, int Nturns=1, int start=0);

    AcceleratorNode* GetLattice();
    PAC::Position getOrbit() const {return m_orbit;} 
    PAC::Position& getOrbit() {return m_orbit;} 
    PacTwissData getTwiss(int index);

    AcceleratorPropagator* getPropagator(){ return m_ap;}


    //correction methods.  Interface same as TEAPOT/api/Teapot/main.pm -
    // so like normal shell.
    
    void hsteer(const char *adjusters, const char *detectors){steer(adjusters, detectors, 'h');}
    void vsteer(const char *adjusters, const char *detectors){steer(adjusters, detectors, 'v');}
    
    void tunethin(const char *focus, const char *defocus, double mux, double muy, char method = '*', 
	        int numtries = 100, double tolerance = 1.e-6, double stepsize = 0.0);
	
    void chromfit(const char *focus, const char *defocus, double xix, double xiy, char method = '*', 
	        int numtries = 10, double tolerance = 1.e-4, double stepsize = 0.0);

    void decouple(const char *a11, const char *a12, const char *a13, const char *a14, 
 		  const char *focus, const char *defocus, double mux, double muy);
//     void map(int order, const char *filename);
//     void matrix(double delta, const char *filename);
     void analysis(const char *filename, double delta=0); // finds closed orbit and does twiss analysis

    ClassDef(RootShell, 2);

  protected:

    static ZLIB::Space s_space;  
    PAC::BeamAttributes m_ba;
    PAC::Position m_orbit;
    AcceleratorPropagator* m_ap; //->
    PacTwissData *m_twiss;//->

#ifndef __CINT__
    //these still need to be hidden....
    PacLattice m_lattice;
    Teapot *m_tea;//->
#endif

    void steer(const char *adjusters, const char *detectors, const char plane);

  };

};


#endif
