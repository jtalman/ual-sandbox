// Library       : SPINK
// File          : SPINK/Propagator/DipoleErTracker.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#ifndef UAL_SPINK_DIPOLEER_TRACKER_HH
#define UAL_SPINK_DIPOLEER_TRACKER_HH

#include "SMF/PacLattElement.h"

#include "SMF/PacElemLength.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacElemMultipole.h"
#include "SMF/PacElemOffset.h"
#include "SMF/PacElemRotation.h"
#include "SMF/PacElemAperture.h"
#include "SMF/PacElemComplexity.h"
#include "SMF/PacElemSolenoid.h"
#include "SMF/PacElemRfCavity.h"

#include "SPINK/Propagator/SpinPropagator.hh"

namespace SPINK {

  /** Basis class of different spin trackers */

  class DipoleErTracker : public SpinPropagator {

  public:

    /** Constructor */
    DipoleErTracker();

   /** Copy constructor */
    DipoleErTracker(const DipoleErTracker& st);

    /** Destructor */
    ~DipoleErTracker();

    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    static void setER(double er) { s_er = er; }

    static double getER() { return s_er; }

  protected:

      void addErKick(PAC::Bunch& bunch);

      static double s_er; // ER(GV/m)/pc(GV)


  protected:

    // Element data

    void setElementData(const PacLattElement& e);

    std::string m_name;

    PacElemMultipole* p_entryMlt;
    PacElemMultipole* p_exitMlt;

    PacElemLength* p_length;           // 1: l
    PacElemBend* p_bend;               // 2: angle, fint
    PacElemMultipole* p_mlt;           // 3: kl, ktl
    PacElemOffset* p_offset;           // 4: dx, dy, ds
    PacElemRotation* p_rotation;       // 5: dphi, dtheta, tilt
    // PacElemAperture*p_aperture;     // 6: shape, xsize, ysize
    PacElemComplexity* p_complexity;   // 7: n
    // PacElemSolenoid* p_solenoid;    // 8: ks
    // PacElemRfCavity* p_rf;          // 9: volt, lag, harmon

    PAC::Bunch m_bunch1, m_bunch2, m_bunch3;

  protected:

    // TEAPOT trackers

    void setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                int is0, int is1,
                                const UAL::AttributeSet& attSet);

    /** Conventional tracker (body) */
    UAL::PropagatorNodePtr m_tracker;

  protected:

    void propagateSpin(UAL::Probe& b);

    double psp0(Position& p, double v0byc);

  private:

    void copy(const DipoleErTracker& st);

  };

  class DipoleErTrackerRegister
  {
    public:

    DipoleErTrackerRegister();
  };


}

#endif
