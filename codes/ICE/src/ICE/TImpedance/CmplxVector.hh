// Library       : ICE
// File          : ICE/TImpedance/CmplxVector.hh
// Copyright     : see Copyright file
// Author        : M.Blaskiewicz
// C++ version   : A.Shishlo 

#ifndef UAL_ICE_CMPLX_VECTOR_HH
#define UAL_ICE_CMPLX_VECTOR_HH

#include "ICE/Base/Def.hh"

namespace ICE {

  /** The CmplxVector class is a class to operate 
   * with vectors of complex variable satisfying 
   * phasor condition.
   */
    
  class CmplxVector
  {
  public:

    /** Constructor */
    CmplxVector(int size);

    /** Destructor */
    virtual ~CmplxVector();

    /** Set the range of operation */
    void  setRange(int i_min, int i_max);

    /** Returns the min range of operation */
    int  getMinRange();

    /** Returns the max range of operation */
    int  getMaxRange();

    /** Sets the real and imaginary parts equal to zero */
    void  zero();

    /** Sets the real part of one element */
    void  setRe(int j, double value);

    /** Sets the imaginary part of one element */
    void  setIm(int j, double value);

    /** Gets the real part of one element */
    double getRe(int j);

    /** Gets the imaginary part of one element */
    double getIm(int j);

    /** Sum two complex vectors */
    void  sum( CmplxVector& cv );

    /** Returns the real part of the sum all components of the complex vector */
    double sumRe();

    /** Returns the Im part of the sum all components of the complex vector */
    double sumIm();

    /** Multiply two complex vectors */
    void  mult( CmplxVector& cv );

    /** Multiply by real value */
    void  multR( double x );

    /** Copy operator */
    void  copy( CmplxVector& cv );

    /** Defines this complex vector as shift one (exp(eta*time)) */
    void  defShift( CmplxVector& eta , double time_shift );

    /** Shifts this complex vector by Multiplying by (exp(eta*time)) */
    void  shift(CmplxVector& eta , double time_shift );

    /** Print vector */
    void  print_();

  private:

    // Size of the complex vector
    int size_;

    // Re and Im values of complex vector
    double* re_;
    double* im_;

  private:

    // Range for operation (static)
    int i_min_;
    int i_max_;
  
  };

}

#endif
