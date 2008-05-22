#ifndef UAL_MIA_HBPM_HH
#define UAL_MIA_HBPM_HH

#include "UAL/APF/PropagatorNode.hh"
#include "SMF/PacLattElement.h"
#include "BPM.hh"

namespace MIA {

  /** BPM for Model Independent Analysis */

  class HBpm : public BPM {

  public:

    /** Constructor */
    HBpm();

    /** Copy constructor */
    HBpm(const HBpm& bpm);

    /** Destructor */
    virtual ~HBpm();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();  

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);
  

    /** Write data */
    void write(std::ofstream& out);
  };

  class HBpmRegister 
  {
    public:

    HBpmRegister(); 
  };


};

#endif
