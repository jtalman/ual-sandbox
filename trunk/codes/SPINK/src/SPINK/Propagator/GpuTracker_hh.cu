#ifndef UAL_SPINK_GPUDIPOLE_TRACKER_HH
#define UAL_SPINK_GPUDIPOLE_TRACKER_HH
#define  precision double
#define PARTICLES 10010
//#define ELEMENTS 5000
#include "SPINK/Propagator/DipoleTracker.hh"
#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
typedef struct {
 precision x, px, y, py, ct, de;
 precision sx, sy, sz;
} vec6D;

//typedef struct {
//precision mlt0, mlt1, mlt2, mlt3, m_ir;
//precision cphpl, sphpl,  tphpl,  scrx, scrs, spxt,  rlipl;
//precision m_l,  kl1, angle, btw01, btw00, atw01, atw00,  dx, dy;
//int type;
//}lattice ;

vec6D pos[PARTICLES];
//lattice rhic[ELEMENTS];

__device__ vec6D pos_d[PARTICLES];
__device__ vec6D tmp_d[PARTICLES];

namespace SPINK {

  class GpuTracker : public DipoleTracker {

  public:

    GpuTracker();

    GpuTracker(const GpuTracker& st);

    /** Destructor **/
    ~GpuTracker();


 /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);



   inline TEAPOT::DipoleData& getDipoleData();

   inline TEAPOT::MagnetData& getMagnetData();


    static precision snk1_mu;
    static precision snk2_mu;
    static precision snk1_phi;
    static precision snk2_phi;
    static precision snk1_theta;
    static precision snk2_theta;

      /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);
 static void setSnakeParams(precision mu1, precision mu2, precision phi1, precision phi2, precision the1, precision the2)
    {snk1_mu = mu1; snk2_mu = mu2; snk1_phi = phi1; snk2_phi = phi2; snk1_theta = the1; snk2_theta = the2;}
    void propagate(UAL::Probe& bunch);
    void BendProp(PAC::Bunch& bunch);
    void MultProp(PAC::Bunch& bunch);
    void RFProp(PAC::Bunch& bunch);
    void DriftProp(PAC::Bunch& bunch);
    void propagateSpin(UAL::Probe& bunch);
    void SnakeProp(PAC::Bunch& bunch);

    UAL::PropagatorNode* clone();

    /** Setup a dump flag for diagnostics AUL:02MAR10 */
    static void setOutputDump(bool outdmp){coutdmp = outdmp;}
    static bool coutdmp;  

      /** Pass information on turn number for diagnostics AUL:02MAR10 */
    static void setNturns(int iturn){nturn = iturn;}
    static int nturn ;
    static void readPart(PAC::Bunch& bunch);
    void loadPart(PAC::Bunch& bunch);

    /** Sets Rf patameters */
    static void setRF(precision V, precision harmon, precision lag) { 
      m_V = V; m_h = harmon; m_lag = lag;
    }

  /** Pass ring length AUL:17MAR10 */
    static void setCircum(precision circum){circ = circum;}
    static precision circ ;



  protected:

    void setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                int is0, int is1,
                                const UAL::AttributeSet& attSet);

 /** Complexity number */
    precision m_ir;

 /** Dipole attributes */
    TEAPOT::DipoleData m_data;

    /** Magnet attributes */
    TEAPOT::MagnetData m_mdata;

   /** Peak RF voltage [GeV] */ 
    static precision m_V;

    /** Phase lag in multiples of 2 pi */
    static precision m_lag;

    /** Harmonic number */
    static precision m_h;

    private:

    void copy(const GpuTracker& st);

  };

  inline TEAPOT::DipoleData& GpuTracker::getDipoleData()
  {
      return m_data;
  }

  inline TEAPOT::MagnetData& GpuTracker::getMagnetData()
  {
      return m_mdata;
  }



 class GpuTrackerRegister
  {
    public:

    GpuTrackerRegister();
  };


}

#endif
