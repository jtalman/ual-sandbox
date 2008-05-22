#ifndef UAL_TEAPOT_DIPOLE_STRING_SC_KICK_HH
#define UAL_TEAPOT_DIPOLE_STRING_SC_KICK_HH

#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/DipoleAlgorithm.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"


namespace TEAPOT {

  /** Composite propagator with sbend element tracker and string space charge kick */

  class DipoleStringSCKick : public BasicTracker {

  public:

    /** Constructor */
    DipoleStringSCKick();

    /** Copy Constructor */
    DipoleStringSCKick(const DipoleStringSCKick& c);

    /** Destructor */
    virtual ~DipoleStringSCKick();

    /** Returns a deep copy of this object (PropagatorNode method) */
    UAL::PropagatorNode* clone();  

    /** Defines the lattice elemements (PropagatorNode method)
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch (PropagatorNode method) */
    void propagate(UAL::Probe& bunch);

  protected:

    void setLatticeElement(const PacLattElement& e);
    void setMagnetData(TEAPOT::MagnetData& md, const PacLattElement& e);
    void propagateSimpleElement(PAC::Bunch& bunch, double v0byc);
    void propagateComplexElement(PAC::Bunch& bunch, double v0byc);


  protected:

    /** Dipole attributes */
    DipoleData m_data;

    /** Magnet attributes */
    MagnetData m_mdata;

   /** Propagator algorithm */
    static DipoleAlgorithm<double, PAC::Position> s_algorithm;

    double m_ir;
    double m_l, m_angle;
    double m_ke1, m_ke2;
  };


  class DipoleStringSCKickRegister 
  {
    public:

    DipoleStringSCKickRegister(); 
  };

};

#endif
