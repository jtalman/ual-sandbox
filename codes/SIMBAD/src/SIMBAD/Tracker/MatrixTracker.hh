// Library       : SIMBAD
// File          : SIMBAD/Tracker/MatrixTracker.hh
// Copyright     : see Copyright file



#ifndef UAL_SIMBAD_MATRIX_TRACKER_HH
#define UAL_SIMBAD_MATRIX_TRACKER_HH

#include <iostream>

#include "SMF/PacLattice.h"
#include "Optics/PacTMap.h"
#include "Main/Teapot.h"
#include "SIMBAD/Tracker/BasicTracker.hh"

namespace SIMBAD {

  /** Matrix tracker. */

  class MatrixTracker :  public BasicTracker {

  public:

    /** Constructor */
    MatrixTracker();

    /** Copy Constructor */
    MatrixTracker(const MatrixTracker& st);

    /** Destructor */
    ~MatrixTracker();

    /** Copy operator */
    const MatrixTracker& operator=(const MatrixTracker& st);

    /** Returns a deep copy of this object (inherited from UAL::PropagatorNode) */
    UAL::PropagatorNode* clone();

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Returns the first node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getBackAcceleratorNode();

    /** Propagates bunch (inherited from UAL::PropagatorNode) */
    void propagate(UAL::Probe& probe);

    void printMatrix(std::ostream& out);

  protected:

    /** Defines matrix data */
    void setMap(const PacVTps& vtps);

  protected:

    /**  Front node */
    PacLattElement m_frontNode;

    /** Back node */
    PacLattElement m_backNode;

  private:

    static PacLattice s_lattice;
    static Teapot     s_teapot;

    double m_l;

    double a10, a11, a12, a13, a14, a15, a16;
    double a20, a21, a22, a23, a24, a25, a26;
    double a30, a31, a32, a33, a34, a35, a36;
    double a40, a41, a42, a43, a44, a45, a46;   
    double a50, a51, a52, a53, a54, a55, a56;
    double a60, a61, a62, a63, a64, a65, a66;  

  private:

    void init();
    void copy(const MatrixTracker& st);

  };

  class MatrixTrackerRegister 
  {
    public:

    MatrixTrackerRegister(); 
  };

}

#endif
