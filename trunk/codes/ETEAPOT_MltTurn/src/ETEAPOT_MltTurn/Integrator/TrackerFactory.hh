#ifndef ETEAPOT_MltTurn_TRACKER_FACTORY_MLT_TURN_HH
#define ETEAPOT_MltTurn_TRACKER_FACTORY_MLT_TURN_HH

#include <map>

#include "UAL/APF/PropagatorNodePtr.hh"
//#include "ETEAPOT_MltTurn/Integrator/BasicTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/MarkerTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DriftTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DipoleTracker.hh"

#include "ETEAPOT_MltTurn/Integrator/MltTracker.hh"
// #include "EETEAPOT_MltTurn/Integrator/MatrixTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/RFCavityTracker.hh"

namespace ETEAPOT_MltTurn {

  /** Factory of the ETEAPOT_MltTurn Trackers */

  class TrackerFactory {

  public:

    /** Returns singleton */
    static TrackerFactory* getInstance();

    /** Returns the tracker specified by the element type */
    static UAL::PropagatorNode* createTracker(const std::string& type);

    /** Returns the default tracker */
    static ETEAPOT::BasicTracker* createDefaultTracker();

    /** Returns the marker tracker */
    static MarkerTracker* createMarkerTracker();

    /** Returns the drift tracker */
    static DriftTracker* createDriftTracker();

    /** Returns the dipole tracker */
    static DipoleTracker* createDipoleTracker();

    /** Returns the multipole tracker */
    static MltTracker* createMltTracker();

    /** Returns the matrix tracker */
    // static MatrixTracker* createMatrixTracker();

  private:

    static TrackerFactory* s_theInstance;
    std::map<std::string, UAL::PropagatorNodePtr> m_trackers;

  private:

    // constructor
    TrackerFactory();

  };

  class TrackerRegister 
  {
    public:

    TrackerRegister(); 
  };


}

#endif
