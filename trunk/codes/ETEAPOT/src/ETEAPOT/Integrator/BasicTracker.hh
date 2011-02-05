// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/BasicTracker.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_BASIC_TRACKER_HH
#define ETEAPOT_BASIC_TRACKER_HH

#include <string>

#include "UAL/Common/Def.hh"
#include "SMF/PacLattElement.h"
#include "PAC/Beam/Position.hh"
#include "SMF/PacElemAperture.h"
#include "SMF/PacElemOffset.h"

#include "ETEAPOT/Integrator/BasicPropagator.hh"

namespace ETEAPOT {

  /** A root class of ETEAPOT conventional integrators. */

  class BasicTracker : public ETEAPOT::BasicPropagator {

  public:

    /** Constructor */
    BasicTracker();

    /** Copy constructor */
    BasicTracker(const BasicTracker& bt);

    /** Destructor */
    virtual ~BasicTracker();

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    // UAL::PropagatorNode* clone();

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    double getLength() { return m_l; }
    int getN() { return m_n; }

  protected:

    /** i0 */
    int m_i0;
    
    /** i1 */
    int m_i1;

    /** Element length */
    double m_l;

    /** slicing number */
    int m_n;

    /** Element s coordinate */
    float m_s;
    
    /** Element name */
    std::string m_name;

    /** Aperture */
    PacElemAperture* m_aperture; 

    /** Offset */
    PacElemOffset* m_offset;

    static double s_maxR;

  protected:

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

    /** Checks aperture */
    void checkAperture(PAC::Bunch& bunch);

  private:

    void initialize();
    void copy(const BasicTracker& bt);

    bool isOK(PAC::Position& pos);
    
  };

}

#endif
