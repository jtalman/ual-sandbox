// Library     : PAC
// File        : Optics/LinearMapper.hh
// Copyright   : see Copyright file

#ifndef UAL_PAC_LINEAR_MAPPER_HH
#define UAL_PAC_LINEAR_MAPPER_HH

#include <iostream>
#include "UAL/APF/PropagatorComponent.hh"
#include "Optics/PacVTps.h"

namespace PAC {

  /** A fast bunch tracker based on the linear matrix approach.
   */

  class LinearMapper : public UAL::PropagatorComponent
  {

  public :

    /** Constructor */
    LinearMapper();

    /** Copy constructor */
    LinearMapper(const LinearMapper& rhs);

    /** Destructor */
    virtual ~LinearMapper();

    /** Creates and returns a copy of this object */
    // PAC::Algorithm* clone();

    /** Defines matrix data */
    void setMap(const PacVTps& vtps);
  
    /** Propagates bunch */
    void propagate(UAL::Probe& bunch);

  protected:

    double a10, a11, a12, a13, a14, a15, a16;
    double a20, a21, a22, a23, a24, a25, a26;
    double a30, a31, a32, a33, a34, a35, a36;
    double a40, a41, a42, a43, a44, a45, a46;   
    double a50, a51, a52, a53, a54, a55, a56;
    double a60, a61, a62, a63, a64, a65, a66;  

  private:

    void init();
    void init(const LinearMapper& rhs);

  };
};

#endif
