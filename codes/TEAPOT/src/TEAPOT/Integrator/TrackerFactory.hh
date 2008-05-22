// Library       : TEAPOT
// File          : TEAPOT/Integrator/TrackerFactory.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_TRACKER_FACTORY_HH
#define UAL_TEAPOT_TRACKER_FACTORY_HH

#include <map>

#include "UAL/APF/PropagatorNodePtr.hh"
#include "TEAPOT/Integrator/DriftTracker.hh"
#include "TEAPOT/Integrator/DipoleTracker.hh"
#include "TEAPOT/Integrator/MltTracker.hh"
#include "TEAPOT/Integrator/MatrixTracker.hh"
#include "TEAPOT/Integrator/RFCavityTracker.hh"

namespace TEAPOT {

  /** Factory of the TEAPOT Trackers */

  class TrackerFactory {

  public:

    /** Returns singleton */
    static TrackerFactory* getInstance();

    /** Returns the tracker specified by the element type */
    static UAL::PropagatorNode* createTracker(const std::string& type);

    /** Returns the default tracker */
    static BasicTracker* createDefaultTracker();

    /** Returns the drift tracker */
    static DriftTracker* createDriftTracker();

    /** Returns the dipole tracker */
    static DipoleTracker* createDipoleTracker();

    /** Returns the multipole tracker */
    static MltTracker* createMltTracker();

    /** Returns the matrix tracker */
    static MatrixTracker* createMatrixTracker();

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
