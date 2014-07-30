#ifndef ETEAPOT_DIPOLE_TRACKER_MLT_TURN_HH
#define ETEAPOT_DIPOLE_TRACKER_MLT_TURN_HH

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "ETEAPOT/Integrator/DipoleData.hh"
#include "ETEAPOT/Integrator/MltData.hh"
//#include "ETEAPOT/Integrator/algorithm.hh"
#include "ETEAPOT/Integrator/BasicTracker.hh"

#include "ETEAPOT_MltTurn/Integrator/DipoleAlgorithm.hh"

#define MAXSXF 1000

namespace ETEAPOT_MltTurn {

  /** bend tracker. */

  class DipoleTracker : public ETEAPOT::BasicTracker {

  public:

    /** Constructor */
    DipoleTracker();

    /** Copy constructor */
    DipoleTracker(const DipoleTracker& dt);

    /** Destructor */
    ~DipoleTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();


    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    inline ETEAPOT::DipoleData& getDipoleData();

    inline ETEAPOT::MltData& getElectricData();

    static double dZFF;

    static double m_m;

    static int bend;

    static double spin[41][3];

    static void initialize(){
 
     ETEAPOT_MltTurn::DipoleTracker::s_algorithm.dZFF=ETEAPOT_MltTurn::DipoleTracker::dZFF;

     char * S[41] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

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
//#include"setDipoleTrackerSpin"
for(int ip=0;ip<=40;ip++){
 for(int iq=0;iq<=2;iq++){
  ETEAPOT_MltTurn::DipoleTracker::spin[ip][iq]=spin[ip][iq];
//((ETEAPOT_MltTurn::DipoleTracker).s_algorithm).spin[ip][iq]=spin[ip][iq];
  ETEAPOT_MltTurn::DipoleTracker::s_algorithm.spin[ip][iq]=spin[ip][iq];
 }
}
    }

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    /** bend attributes */
    ETEAPOT::DipoleData m_data;

    /** Electric attributes */
    ETEAPOT::MltData m_edata;

  public:

    /** Propagator algorithm */
    static DipoleAlgorithm<double, PAC::Position> s_algorithm;

  };

  inline ETEAPOT::DipoleData& DipoleTracker::getDipoleData()
  {
      return m_data;
  }

  inline ETEAPOT::MltData& DipoleTracker::getElectricData()
  {
      return m_edata;
  }

}

#endif
