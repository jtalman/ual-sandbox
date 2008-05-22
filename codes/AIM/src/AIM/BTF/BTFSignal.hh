// Library       : AIM
// File          : AIM/BTF/BTFSignal.hh
// Copyright     : see Copyright file

#ifndef UAL_AIM_BTF_SIGNAL_HH
#define UAL_AIM_BTF_SIGNAL_HH

#include <vector>

namespace AIM {

  /** BPM signal */

  class BTFSignal {

  public:

    /** Constructor */
    BTFSignal();

    /** Resizes containers */
    void resize(int size);

    /** Longitudinal positions */
    std::vector<double> cts;

    /** Line density */
    std::vector<double> density;

    /** Horizontal dipole driving terms */
    std::vector<double> xs;

    /** Vertical dipole driving terms */
    std::vector<double> ys;    
  };

}

#endif
