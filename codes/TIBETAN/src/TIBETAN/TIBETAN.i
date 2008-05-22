%inline %{
#include <assert.h>
%}

%module TIBETAN

%include "typemaps.i"

%{
#include "TIBETAN/Propagator/RFCavityTracker.hh"
%}

namespace TIBETAN {

  /** RF Cavity Tracker */

  class RFCavityTracker : public BasicPropagator {

  public:

    /** Constructor */
    RFCavityTracker();

    /** Destructor */
    virtual ~RFCavityTracker();

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);
    
  };

};
