#ifndef ETEAPOT_DRIFT_ALGORITHM_MLT_TURN_HH
#define ETEAPOT_DRIFT_ALGORITHM_MLT_TURN_HH

#include <iomanip>
#include "ETEAPOT/Integrator/CommonAlgorithm.hh"

namespace ETEAPOT_MltTurn {
 
  /** A template of the drift algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DriftAlgorithm 
    : public ETEAPOT::CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DriftAlgorithm();

    /** Destructor */
    ~DriftAlgorithm();

    void passDriftPlusPostProcess(
      int ip,
      double rlipl, 
      Coordinates& p,  
      Coordinates& tmp, 
      double v0byc,
      double m_m,
      int drft);

  };

}

#include "ETEAPOT_MltTurn/Integrator/DriftAlgorithm.icc"

#endif
