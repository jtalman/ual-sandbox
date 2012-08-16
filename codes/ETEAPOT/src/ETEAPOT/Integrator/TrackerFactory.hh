// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/TrackerFactory.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_TRACKER_FACTORY_HH
#define ETEAPOT_TRACKER_FACTORY_HH

#include <map>

#include "UAL/APF/PropagatorNodePtr.hh"
#include "ETEAPOT/Integrator/MarkerTracker.hh"
#include "ETEAPOT/Integrator/DriftTracker.hh"
#include "ETEAPOT/Integrator/DipoleTracker.hh"

#include "ETEAPOT/Integrator/MltTracker.hh"
// #include "EETEAPOT/Integrator/MatrixTracker.hh"
#include "ETEAPOT/Integrator/RFCavityTracker.hh"

namespace ETEAPOT {

  /** Factory of the ETEAPOT Trackers */

  class TrackerFactory {

  public:

    /** Returns singleton */
    static TrackerFactory* getInstance();

    /** Returns the tracker specified by the element type */
    static UAL::PropagatorNode* createTracker(const std::string& type);

    /** Returns the default tracker */
    static BasicTracker* createDefaultTracker();

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
