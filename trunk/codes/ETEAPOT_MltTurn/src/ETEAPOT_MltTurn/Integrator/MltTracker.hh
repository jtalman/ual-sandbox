#ifndef ETEAPOT_MLT_TRACKER_MLT_TURN_HH
#define ETEAPOT_MLT_TRACKER_MLT_TURN_HH

#include <stdlib.h>
#include "SMF/PacElemMultipole.h"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#include "ETEAPOT_MltTurn/Integrator/MltAlgorithm.hh"

namespace ETEAPOT_MltTurn {

  /** Multipole Tracker. */

  class MltTracker : public ETEAPOT::BasicTracker {

  public:

    /** Constructor */
    MltTracker();

    /** Copy constructor */
    MltTracker(const MltTracker& mt);

    /** Destructor */
    ~MltTracker();

//  friend class DipoleTracker;

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);


    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    inline ETEAPOT::MltData& getMltData();

    static double m_m;

    static int mltK;

    static double spin[41][3];

static void initialize(){
 std::cerr << "enter static void initialize(){ \n";
#include"S"
// char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

 ifstream spinIFS;
 spinIFS.open ("initialSpin", ifstream::in);


 PAC::Spin echo;
 std::string spinX,spinY,spinZ;
 int ip=-1;
 std::string Name;
 double spin[41][3];
 while(1){
  ip++;
  spinIFS >> Name >> spinX >> spinY >> spinZ;
  echo.setSX( atof(spinX.c_str()) );echo.setSY( atof(spinY.c_str()) );echo.setSZ( atof(spinZ.c_str()) );

  if( !spinIFS.eof() ){
   spin[ip][0]=echo.getSX();spin[ip][1]=echo.getSY();spin[ip][2]=echo.getSZ();
  }
  else{
   spin[ip][0]=echo.getSX();spin[ip][1]=echo.getSY();spin[ip][2]=echo.getSZ();
   break;
  }
 }

 spinIFS.close();

//#include"setMltTrackerSpin"
for(int ip=0;ip<=40;ip++){
 for(int iq=0;iq<=2;iq++){
//ETEAPOT_MltTurn::MltTracker::spin[ip][iq]=spin[ip][iq];
  ETEAPOT_MltTurn::MltTracker::s_algorithm.spin[ip][iq]=spin[ip][iq];
 }
}
}

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** Element length */
    // double m_l;

    /** Complexity number */
    double m_ir;

    /** Mlt attributes */
    ETEAPOT::MltData m_mdata;

  public:

    /** Propagator algorithm */
    static ETEAPOT_MltTurn::MltAlgorithm<double, PAC::Position> s_algorithm;

  private:

//  void initialize();
    void copy(const MltTracker& mt);

  };

  inline ETEAPOT::MltData& MltTracker::getMltData()
  {
      return m_mdata;
  }

}

#endif
