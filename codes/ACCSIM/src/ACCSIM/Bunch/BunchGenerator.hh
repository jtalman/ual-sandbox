//# Library       : ACCSIM
//# File          : ACCSIM/Bunch/BunchGenerator.hh
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_BUNCH_GENERATOR_HH
#define UAL_ACCSIM_BUNCH_GENERATOR_HH

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTwissData.h"
#include "ACCSIM/Base/Def.hh"
#include "ACCSIM/Base/UniformGenerator.hh"
#include "ACCSIM/Base/TeapotGenerator.hh"

namespace ACCSIM {

  /**
     A collection of algorithms for producing several beam distributions
     with different phase space boundaries (rectangle or ellipse) and 
     profiles (uniform, binomial, or gaussian).

     Rectanlge sizes are defined by the halfWidth object,
     an instance of the PacPosition class:
     <ul>
     <li> halfWidth.x()   - half-width in x  direction [m] 
     <li> halfWidth.px()  - half-width in px direction [1] 
     <li>...
     </ul>

     Ellipse sizes can be defined by the halfWidth object
     (as for rectangle) or by "twiss" and "emittance" ("emittance" 
     ellipse contains 100% of the beam):
     <ul>
     <li> twiss.beta(0)  - x beta [m]
     <li> twiss.alpha(0) - x alpha [1]
     <li> twiss.beta(1)  - y beta [m]
     <li> twiss.alpha(1) - y alpha [1]
     <li> emittance.x()  - horizontal emittance [m]
     <li> emittance.y()  - vertical emittance [m]
     <li> emittance.ct() - bunch length [m]
     <li> emittance.de() - relative energy spread [1]
     </ul>

     Binomial distribution is parameterized by parameter m:
     <ul>
     <li> 0.0 - Hollow shell
     <li> 0.5 - Flat profile
     <li> 1.0 - Uniform (elliptical profile)
     <li> 1.5 - Elliptical (parabolic profile)
     <li> 2.0 - Parabolic
     <li> infinity - Gaussian
     </ul>
 
  */


  class BunchGenerator 
  {

  public:

    /** Constructor */
    BunchGenerator();  

    /** Destructor */
    virtual ~BunchGenerator();

    /** Shifts bunch particles (injection bumps) */
    void shift(PAC::Bunch& bunch, PAC::Position& kick);

    /** Updates the bunch distribution by uniformly populated rectangles.*/
    void addUniformRectangles(PAC::Bunch& bunch,
			      const PAC::Position& halfWidth,
			      int& seed);

    /** Updates the bunch distribution by "gaussian" rectangles. */
    void addGaussianRectangles(PAC::Bunch& bunch,
			       const PAC::Position& rms,
			       double cut,
			       int& seed);

    /** Updates the bunch distribution by uniformly populated ellipses (not implemented yet). */
    void addUniformEllipses(PAC::Bunch& bunch,
			    const PAC::Position& halfWidth,
			    int& seed);

    /** Updates the bunch distribution by  uniformly populated ellipses(not implemented yet). */
    void addUniformEllipses(PAC::Bunch& bunch,
			    const PacTwissData& twiss,
			    const PAC::Position& emittance,
			    int& seed);


    /** Updates the bunch distribution by  ellipses with binominal distribution. */
    void addBinomialEllipses(PAC::Bunch& bunch,
			     double m,
			     const PacTwissData& twiss,
			     const PAC::Position& emittance,
			     int& seed);

    // Add a binominal distribution in the index1/index2 plane. 
    void addBinomialEllipse1D(/*inout*/ PAC::Bunch& bunch,
			    /*in*/ double m,
			    /*in*/ int index1,
			    /*in*/ double halfWidth1,
			    /*in*/ int index2,
			    /*in*/ double halfWidth2,
			    /*in*/ double alpha,
			    /*inout*/ int& seed);

  private:

    // Invoke local random number generator
    double uran(int& seed);

  private:

    ACCSIM::UniformGenerator   uniformGenerator_;

  };
}

#endif



 
