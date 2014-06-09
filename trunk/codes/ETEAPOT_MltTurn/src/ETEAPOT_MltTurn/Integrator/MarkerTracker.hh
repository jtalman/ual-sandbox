#ifndef ETEAPOT_MARKER_TRACKER_MLT_TURN_HH
#define ETEAPOT_MARKER_TRACKER_MLT_TURN_HH

#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<iomanip>
#include "ETEAPOT/Integrator/BasicTracker.hh"

namespace ETEAPOT_MltTurn {

  /** Marker tracker. */

  class MarkerTracker : public ETEAPOT::BasicTracker {

  public:

    /** Constructor */
    MarkerTracker();

    /** Copy constructor */
    MarkerTracker(const MarkerTracker& dt);

    /** Destructor */
    ~MarkerTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    static int mark;
    static std::string Mark_m_elementName[1000];
    static double Mark_m_sX[1000];

    static double spin[21][3];

  protected:

    // Sets the lattice element 
    // void setLatticeElement(const PacLattElement& e);

  protected:

    /** Propagator algorithm */
//  static MarkerAlgorithm<double, PAC::Position> s_algorithm;

  private:
    // void copy(const MarkerTracker& dt);

  public:

/*
static void initialize(){
 char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};
#include"setMarkerTrackerSpin"
}
*/

static void initialize(){
 std::cerr << "enter static void initialize(){ \n";
 char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

 ifstream spinIFS;
 spinIFS.open ("initialSpin", ifstream::in);

/*
 ofstream spinOFS;
   spinOFS.open ("out/VERIF/markerSpin");
// spinOFS.open ("out/VERIF/markerSpin_SrvSd");
 spinOFS << setiosflags( ios::showpos    );  
 spinOFS << setiosflags( ios::uppercase  );  
 spinOFS << setiosflags( ios::scientific );
 spinOFS << setfill( ' ' );
 spinOFS << setiosflags( ios::left );
 spinOFS << setprecision(13) ;
*/

 PAC::Spin echo;
 std::string spinX,spinY,spinZ;
 int ip=-1;
 std::string Name;
 double spin[21][3];
 while(1){
  ip++;
  spinIFS >> Name >> spinX >> spinY >> spinZ;
  echo.setSX( atof(spinX.c_str()) );echo.setSY( atof(spinY.c_str()) );echo.setSZ( atof(spinZ.c_str()) );

  if( !spinIFS.eof() ){
// spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ() << "\n";
   spin[ip][0]=echo.getSX();spin[ip][1]=echo.getSY();spin[ip][2]=echo.getSZ();
  }
  else{
// spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ();
   spin[ip][0]=echo.getSX();spin[ip][1]=echo.getSY();spin[ip][2]=echo.getSZ();
   break;
  }
 }

 spinIFS.close();
// spinOFS.close();

#include"setMarkerTrackerSpin"
}

  };

}

#endif
