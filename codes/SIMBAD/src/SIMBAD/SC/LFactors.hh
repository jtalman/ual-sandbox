// Library       : SIMBAD
// File          : SIMBAD/SC/LFactors.hh
// Copyright     : see Copyright file
// Author        : N.Malitsky 

#ifndef UAL_SIMBAD_LFACTORS_HH
#define UAL_SIMBAD_LFACTORS_HH

#include <vector>
#include "PAC/Beam/Bunch.hh"

namespace SIMBAD {

  /** The LFactors class is a global container of longitudinal position 
      factors shared by longitudinal and transverse space charge integrators.
  */
    
  class LFactors
  {
  public:

    /** Get the instance (singleton) */
    static LFactors& getInstance(int  maxBunchSize);

    /**  Return the container size */
    int getSize() const;

    /** Return the factor value */
    double getElement(int index) const; 

    /** Set the factor value */
    void setElement(double value, int index);

  protected:

    /** Constructor */
    LFactors(int maxBunchSize);

  protected:

    static LFactors* s_theInstance;

    std::vector<double> m_lFactors;
   
  };

}

#endif
