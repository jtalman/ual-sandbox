#ifndef ETEAPOT2_TRACKER_FACTORY_HH
#define ETEAPOT2_TRACKER_FACTORY_HH

#include <map>

#include "UAL/APF/PropagatorNodePtr.hh"

#include "ETEAPOT2/Integrator/drift.cc"
#include "ETEAPOT2/Integrator/bend.cc"
#include "ETEAPOT2/Integrator/marker.cc"
#include "ETEAPOT2/Integrator/quad.cc"
#include "ETEAPOT2/Integrator/sext.cc"
#include "ETEAPOT2/Integrator/oct.cc"
#include "ETEAPOT2/Integrator/rfCavity.cc"

namespace{
#include "ETEAPOT2/Integrator/genMethods/spinExtern"
#include "ETEAPOT2/Integrator/genMethods/designExtern"
#include "ETEAPOT2/Integrator/genMethods/bunchParticleExtern"
}

namespace ETEAPOT2 {

  /** Factory of the ETEAPOT2 Trackers */

  class TrackerFactory {

  public:

    /** Returns singleton */
    static TrackerFactory* getInstance();

    /** Returns the tracker specified by the element type */
    static UAL::PropagatorNode* createTracker(const std::string& type);

    /** Returns the default tracker */
    static ETEAPOT::BasicTracker* createDefaultTracker();

    /** Returns the marker tracker */
    static marker* createmarker();

    /** Returns the drift tracker */
    static drift* createdrift();

    /** Returns the dipole tracker */
    static bend* createbend();

    static quad* createquad();
    static sext* createsext();
    static oct* createoct();

    /** Returns the rf cavity tracker */
    static rfCavity* createrfCavity();

    /** Returns the matrix tracker */
//  static MatrixTracker* createMatrixTracker();

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
