// Library       : ICE
// File          : ICE/TImpedance/TImpedanceWF.hh
// Copyright     : see Copyright file
// Author        : M.Blaskiewicz
// C++ version   : A.Shishlo 

#ifndef UAL_ICE_TIMPEDANCE_WF_HH
#define UAL_ICE_TIMPEDANCE_WF_HH

#include "PAC/Beam/Bunch.hh"
#include "ICE/TImpedance/CmplxVector.hh"

namespace ICE {

  /** The  TImpedanceWF class uses the wake fields (WF) 
   * for the transverse impedance implementation.
   */
    
  class TImpedanceWF
  {
  public:

    /** Constructor */
    TImpedanceWF(int nBins, int nImpElements, int maxBunchSize);

    /** Destructor */
    virtual ~TImpedanceWF();

    /** Sets the parameters of the impedance element's wake field */
    void addElement( int i_xy,
		     double wake_zero_re, 
		     double wake_zero_im,
		     double eta_coff_re,
		     double eta_coff_im );

    /** Sets the parameters of the resonant impedance element's wake field.
     *  r  - resistance parameter [ohms/m^2]
     *  q  - quality factor
     *  fr - resonant frequency [Hz]
     */
    void addResonantElement( int i_xy,
			     double r, 
			     double q,
			     double fr);

    /** Sets the range of operation for CmplxVector's arrays.
     * There should be placed all CmplxVector variables.
     */
    void setRange_();

    /** Gets wake function for x or y (i_xy) and an WF element 
     * with index j at the moment of time t [sec].
     */
    double getWF(int i_xy, int j, double t);


    /** Restores initial element's state */
    void restore();

    /** Memorizes the state of the transverse impedance element */
    void memorizeState();

    /** Restores the memorized state of the transverse impedance element */
    void restoreState();

    /** Prints out the <x> and <y> of bunch */
    void showXY(char* f);

    /** Prints out the <x> and <y> momenta of bunch */
    void showTKick(char* f);

    /** Propagates the bunch through the transverse impedance element */
    void propagate(PAC::Bunch& bunch , double t);

    /** Prints the parameters of the impedance element's wake fields */
	void printElementParameters();

  protected:

    /**  Transverse Kick calculation */
    void tKickCalc_( double t);

    /** Get v0/c */
    double getBeta_(const PAC::Bunch& bunch);

    /** Bin the macro particles longitudinally */
    void  defineLongBin(const PAC::Bunch& bunch);

    /** Synchronize longitudinal grid size for a parallel version 
     * (empty method here)
     */
    virtual void _grid_synchronize(double& ct_min, double& ct_max);

    /** Sum longitudinal distribution for a parallel version 
     * (empty method here)
     */
    virtual void _sum_distribution();


  private:

    // coefficient (r0*4*pi/(Z0*beta**2*gamma))
    double coefficient_;

    // time_min_prev_ - minimal time for previous bunch 
    double time_min_prev_;

    // Integrator's containers:

    int nBins_;
    int nElem_;
    int maxBunchSize_;  

    int* nElemExist_;
    int* xy_mask_;

    double**  xyM_;
    double**  tSumKick_;

    CmplxVector***  tKick_ ;

    CmplxVector**  tfKick_ ;

    CmplxVector**  tFFKick_;

    CmplxVector**  tFFmemory_;

    CmplxVector** wake_zero_;

    CmplxVector** eta_fact_;

    CmplxVector** eta_coff_;

    // (size: maxBunchSize_)
    int* bins_;

    // (size: maxBunchSize_)
    double* fractBins_;

    // actual BunchSize
    int bunchSize_;

    // time spread 
    double t_min_,t_max_,t_step_;
 
  };

}

#endif
