// Library       : SPINK
// File          : SPINK/Propagator/GpuTracker_hh.cu
// Copyright     : see Copyright file
// Author        : V.Ranjbar
/** header file for GpuTracker.cu **/

#ifndef UAL_SPINK_GPUDIPOLE_TRACKER_HH
#define UAL_SPINK_GPUDIPOLE_TRACKER_HH
/** set the global precision here **/
#define  precision double
/** set the maximum number of particles to be tracked **/
#define PARTICLES 10000
/** set maximum number of elements to be tracked through **/
#define ELEMENTS 3000
#include "SPINK/Propagator/DipoleTracker.hh"
#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"

// struture for holding particle data for use on GPU
typedef struct {
 precision x, px, y, py, ct, de;
  double sx, sy, sz;
} vec6D;

// structure for holding lattice data for use on GPU

typedef struct {
  // lattice definitions for use in orbit transport
  //multipole transport:
  precision mlt[10], entryMlt[10], exitMlt[10],kl1;
  precision m_l,dx,dy;
  // int MULT, EXIT, ENTRY;
  //bend transport:
  precision angle,btw01,btw00,atw01,atw00;
  precision cphpl[20], sphpl[20],tphpl[20],scrx[20],rlipl[20];
  precision scrs[20],spxt[20];
  //element flags
  int rfcav, snake;  // no rf at element = 0,  rfcavity at element  = 1,
  //  0 no snake, 1 first snake,  2 second snake
  // complexity and order indicators
  int ns, m_ir, order;
  // multipole elements calculated for Spin transport
  precision k1l ,k0l,kls0, k2l,length,bend;
  }Lat;

typedef struct {
  precision mlt[10];}Qlat;


vec6D pos[PARTICLES];
Lat rhic[ELEMENTS];

 precision Energy[PARTICLES];

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
    static precision stepsize;
    static precision TOL;
    static precision dthread;
    static int threads;

      /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);
 static void setSnakeParams(precision mu1, precision mu2, precision phi1, precision phi2, precision the1, precision the2)
    {snk1_mu = mu1; snk2_mu = mu2; snk1_phi = phi1; snk2_phi = phi2; snk1_theta = the1; snk2_theta = the2;}
   
    UAL::PropagatorNode* clone();

    /** Setup a dump flag for diagnostics AUL:02MAR10 */
    static void setOutputDump(bool outdmp){coutdmp = outdmp;}
    static bool coutdmp;  

      /** Pass information on turn number for diagnostics AUL:02MAR10 */
    static void setNturns(int iturn){nturn = iturn;}
    static int nturn ;
    static void readPart(PAC::Bunch& bunch, int printall);
    static void loadPart(PAC::Bunch& bunch);
    static void setStep(precision step){ stepsize = step;};
   

    /** Sets Rf patameters */
    static void setRF(precision V, precision harmon, precision lag) { 
      m_V = V; m_h = harmon; m_lag = lag;
    }

    /** set number of Time parallel segments per GPU **/
    static void setThreads(precision dthread1, int threads1){dthread = dthread1; threads = threads1;
      std::cout << "threads = " << threads << "dthread =" << dthread << " \n";
}

  /** Pass ring length AUL:17MAR10 */
    static void setCircum(precision circum){circ = circum;}
    static precision circ ;

    /** GpuPropagator **/
    static void GpuProp(PAC::Bunch& bunch);

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
     
    static int Nelement;

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
