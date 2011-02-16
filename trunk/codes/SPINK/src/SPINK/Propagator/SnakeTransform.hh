// Library       : SPINK
// File          : SPINK/Propagator/SnakeTransform.hh
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : V.Ptitsyn

#ifndef UAL_SPINK_SNAKE_TRANSFORM_HH
#define UAL_SPINK_SNAKE_TRANSFORM_HH

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

  class SnakeTransform : public SpinPropagator {

  public:

    /** Constructor */
    SnakeTransform();

   /** Copy constructor */
    SnakeTransform(const SnakeTransform& st);

    /** Destructor */
    ~SnakeTransform();
    /** Defines the lattice elemements (PropagatorNode method)
	Note: integers i0 and i1 will be replaced by AcceleratorNode's 
    */
    virtual void setLatticeElements(const UAL::AcceleratorNode& sequence, int i0, int i1, 
				    const UAL::AttributeSet& attSet);

    /** Propagates a bunch */
    void propagate(UAL::Probe& bunch);

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    static void setSnakeParams(double mu1, double mu2, double phi1, double phi2, double the1, double the2)
    {snk1_mu = mu1; snk2_mu = mu2; snk1_phi = phi1; snk2_phi = phi2; snk1_theta = the1; snk2_theta = the2;} //AULNLD 2/9/10
 
    static double snk1_mu;
    static double snk2_mu;
    static double snk1_phi;
    static double snk2_phi;
    static double snk1_theta;
    static double snk2_theta;

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

    void copy(const SnakeTransform& st);
 
  };

  class SnakeTransformRegister
  {
    public:

    SnakeTransformRegister();
  };


}

#endif
