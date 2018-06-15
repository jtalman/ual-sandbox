#ifndef EMTEAPOT_TRACKER_FACTORY_HH
#define EMTEAPOT_TRACKER_FACTORY_HH

#include <map>

#include "UAL/APF/PropagatorNodePtr.hh"

#include "EMTEAPOT/Integrator/drift.cc"
#include "EMTEAPOT/Integrator/embend.cc"
#include "EMTEAPOT/Integrator/marker.cc"
#include "EMTEAPOT/Integrator/quad.cc"
#include "EMTEAPOT/Integrator/sext.cc"
#include "EMTEAPOT/Integrator/oct.cc"
#include "EMTEAPOT/Integrator/rfCavity.cc"

namespace{
#include "EMTEAPOT/Integrator/genMethods/spinExtern"
#include "EMTEAPOT/Integrator/genMethods/designExtern"
#include "EMTEAPOT/Integrator/genMethods/bunchParticleExtern"
}

namespace EMTEAPOT {

  /** Factory of the EMTEAPOT Trackers */

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
    static embend* createembend();

    static quad* createquad();
    static sext* createsext();
    static oct* createoct();

    /** Returns the rf cavity tracker */
    static rfCavity* createrfCavity();

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
