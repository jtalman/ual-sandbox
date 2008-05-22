// Library       : ICE
// File          : ICE/TImpedance/CmplxVector.cc
// Copyright     : see Copyright file
// Author        : M.Blaskiewicz
// C++ version   : A.Shishlo 

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "ICE/TImpedance/CmplxVector.hh"

// Constructor

ICE::CmplxVector::CmplxVector(int size) 
{
  size_ = size;

  re_ = new double[size_];
  im_ = new double[size_];
  
  int i;
  for ( i = 0 ; i < size_ ; i++){
    re_[i]= 0.0;
    im_[i]= 0.0;
  }

  //Set range for operation
  i_min_ = 0;
  i_max_ = size_ -1;

}


// Destructor

ICE::CmplxVector::~CmplxVector()
{
  delete [] re_;
  delete [] im_;
  //cerr << "Destructor was done. CmplxVector  !!!! \n";
}


//Set the range of operation
void ICE::CmplxVector::setRange(int i_min, int i_max)
{
  i_min_ = i_min;
  i_max_ = i_max;
  if(i_min < 0 ) i_min_ = 0;
  if(i_max > (size_ - 1)) i_max_ = size_ - 1;
   if( i_min_ > i_max_ ) {
     std::cout << "CmplxVector::setRange : i_min > i_max - error. Stop.\n";
     exit(1);
   }
}

//Returns the min range of operation
int ICE::CmplxVector::getMinRange()
{
  return i_min_;
}

//Returns the max range of operation
int ICE::CmplxVector::getMaxRange()
{
  return i_max_;
}

//Set the real and imaginary parts equal to zero
void ICE::CmplxVector::zero()
{
  int i;
  for ( i = i_min_ ; i <= i_max_ ; i++){
    re_[i]= 0.0;
    im_[i]= 0.0;
  }
}

//Set the real part of one element
void ICE::CmplxVector::setRe(int j, double value)
{
  re_[j] = value;
}

//Set the imaginary part of one element
void ICE::CmplxVector::setIm(int j, double value)
{
  im_[j] = value;
}

//Get the real part of one element
double ICE::CmplxVector::getRe(int j)
{
  return re_[j];
}

//Get the imaginary part of one element
double ICE::CmplxVector::getIm(int j)
{
  return im_[j];
}

//Retirns the real part of the sum all components of the complex vector
double ICE::CmplxVector::sumRe()
{
  int i;
  double d = 0.0;
  for( i = i_min_ ; i <= i_max_ ; i++){
    d +=re_[i];
  }
  return d;
}

//Retirns the Im part of the sum all components of the complex vector
double ICE::CmplxVector::sumIm()
{
  int i;
  double d = 0.0;
  for( i = i_min_ ; i <= i_max_ ; i++){
    d +=im_[i];
  }
  return d;
}

//Sum two complex vectors
void ICE::CmplxVector::sum( ICE::CmplxVector& cv )
{
  int i;
  for( i = i_min_ ; i <= i_max_ ; i++){
    re_[i] = re_[i] + cv.getRe(i);
    im_[i] = im_[i] + cv.getIm(i);
  }
}

//Multiply two complex vectors
void ICE::CmplxVector::mult( ICE::CmplxVector& cv )
{
  int i;
  double x;
  for( i = i_min_ ; i <= i_max_ ; i++){

    // debugging
    // cerr << "i = " << i << "\n";
    // cerr << "Var 1  re and im ===== " << re_[i] << "  " << im_[i] << "\n";
    // cerr << "Var 2  re and im ===== " << cv.getRe(i) << "  " << cv.getIm(i) << "\n";

    x = re_[i];
    re_[i] = re_[i]*cv.getRe(i) - im_[i]*cv.getIm(i);
    im_[i] = x*cv.getIm(i) + im_[i]*cv.getRe(i);

  }
}

//Multiply by real value
void ICE::CmplxVector::multR( double x )
{
  int i;
  for( i = i_min_ ; i <= i_max_ ; i++){
    re_[i] = re_[i]*x;
    im_[i] = im_[i]*x;
  }
}

//Copy operator 
void ICE::CmplxVector::copy( ICE::CmplxVector& cv )
{
  int i;
  for( i = i_min_ ; i <= i_max_ ; i++){
    re_[i] = cv.getRe(i);
    im_[i] = cv.getIm(i);
  }
}

//Defines this complex vector as shift one (exp(eta*time))
void ICE::CmplxVector::defShift( ICE::CmplxVector& eta , double time_shift )
{
  int i;
  double x1,x2;
  for( i = i_min_ ; i <= i_max_ ; i++){
    x1 = exp(eta.getRe(i)*time_shift);
    x2 = eta.getIm(i)*time_shift;
    re_[i] = x1*cos(x2);
    im_[i] = x1*sin(x2);
  }
}

//Shifts this complex vector by Multiplying by (exp(eta*time))
void ICE::CmplxVector::shift(ICE::CmplxVector& eta , double time_shift )
{
  int i;
  double x,x1,x2, shift_re,shift_im;
  for( i = i_min_ ; i <= i_max_ ; i++){
    x1 = exp(eta.getRe(i)*time_shift);
    x2 = eta.getIm(i)*time_shift;
    shift_re = x1*cos(x2);
    shift_im = x1*sin(x2);
    x = re_[i];
    re_[i] = re_[i]*shift_re - im_[i]*shift_im;
    im_[i] = x*shift_im + im_[i]*shift_re;
  }
}

//Print vector
void ICE::CmplxVector::print_()
{
  int i;
  std::cerr << "==CmplxVector:print Re and Im parts. \n";
  std::cerr << "  range i_min = " << i_min_ <<"  i_max =" << i_max_ <<" \n";

  for( i = i_min_ ; i <= i_max_ ; i++){
    std::cerr << "  index = " << i 
	      << "  Re = " << re_[i] 
	      << "  Im = " << im_[i] << " \n";
  }
  std::cerr << "==CmplxVector:print ====stop=====. \n";  
}
