#ifndef UAL_RFSOLENOID_TRANSFORM_HH
#define UAL_RFSOLENOID_TRANSFORM_HH

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

  class RFSolenoid : public SpinPropagator {

  public:

    /** Constructor */
    RFSolenoid();

   /** Copy constructor */
    RFSolenoid(const RFSolenoid& st);

    /** Destructor */
    ~RFSolenoid();
    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    static void setRFSParams(double b, char r, double f, double d, int n)
    {RFS_Bdl = b; RFS_rot = r; RFS_freq0 = f; RFS_dfreq = d; RFS_nt = n;}
 
    static double RFS_Bdl;
    static char RFS_rot;
    static double RFS_freq0;
    static double RFS_dfreq;
    static int RFS_nt;

     /** Setup a dump flag for diagnostics AUL:01MAR10 */
    static void setOutputDump(bool outdmp){coutdmp = outdmp;}
    static bool coutdmp; 

     /** Pass information on turn number for diagnostics AUL:01MAR10 */
    static void setNturns(int iturn){nturn = iturn;}
    static int nturn ;

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

  protected:

    // TEAPOT trackers

    void setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                int is0, int is1,
                                const UAL::AttributeSet& attSet);

    /** Conventional tracker (body) */
    UAL::PropagatorNodePtr m_tracker;

    //  protected:

  protected:

    void propagateSpin(UAL::Probe& b);
    void propagateSpin(PAC::BeamAttributes& ba, PAC::Particle& prt);

    double get_psp0(PAC::Position& p, double v0byc);


  private:

    void copy(const RFSolenoid& st);
 
  };

  class RFSolenoidRegister
  {
    public:

    RFSolenoidRegister();
  };


}

#endif
