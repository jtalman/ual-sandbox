// Library     : ZLIB
// File        : ZLIB/Tps/Tps.hh
// Description : Truncated power series
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky & Alexander Reshetov
 
#ifndef UAL_ZLIB_TPS_HH
#define UAL_ZLIB_TPS_HH

#include "ZLIB/Tps/Space.hh"

namespace ZLIB {

  class VTps;
  class Tps;

      // Dif. operators

      /** Returns the partial derivative of the zs series with respect to iv-variable.*/
      Tps D(const Tps& zs, unsigned int iv);

      /** Returns the Poisson bracket of zs1 and zs2 series:*/
      Tps poisson(const Tps& zs1, const Tps& zs2);

      // Other operators
      /** Unary minus.*/
      Tps operator-(const Tps& rhs);

      /** Returns the temporary object after addition of the rhs series and constant. */
      Tps operator+(const Tps& rhs, double c);

      /** Returns the temporary object after addition of constant and the rhs series.*/
      Tps operator+(double c, const Tps& rhs);

      /** Returns the temporary object after subtraction of the rhs series and constant. */
      Tps operator-(const Tps& rhs, double c);

      /** Returns the temporary object after subtraction of constant and the rhs series. */
      Tps operator-(double c, const Tps& rhs);

      /** Returns the temporary object after multiplication of  the rhs series and constant. */
      Tps operator*(const Tps& rhs, double c);

      /** Returns the temporary object after multiplication of constant and the rhs series. */
      Tps operator*(double c, const Tps& rhs);

      /** Returns the temporary object after division of the rhs series by constant. */
      Tps operator/(const Tps& rhs, double c);

      /** Returns the temporary object after division of  constant by the rhs series. */
      Tps operator/(double c, const Tps& rhs); 

      // Some useful math. functions

      /** Returns the square root of the zs series. */
      Tps sqrt(const Tps& zs);

  /** Represents the data and algebra of Truncated Power Series (TPS). */

  class Tps : public ZLIB::Space 
    {
      public :

      /** Constructor */
      Tps();

      /** Constructor with two arguments: initial value (c) and order. */
      Tps(double c, unsigned int order);

      /** Copy constructor */
      Tps(const Tps& rhs);

      /** Destructor */
      virtual ~Tps();

      // Attributes 

      /** Returns the order of this object.*/
      unsigned int order() const;

      /** Sets the order of this object. */
      void order(unsigned int o);

      /** Returns the number of monomials */
      unsigned int size() const;

      // Operators

      // Access operators

      /** Returns a reference to the jth coefficient.*/
      double&  operator[](unsigned int index);

      /** Returns a value of the jth coefficient. */
      double   operator[](unsigned int j) const;

      // Assignment operators

      /** Sets the first coefficient and removes higher order monomials */
      Tps& operator =(double c);

      /** Adds scalar and assigns. */
      Tps& operator+=(double c);

      /** Subtracts scalar and assigns. */
      Tps& operator-=(double c);

      /** Multiplies by scalar and assigns. */
      Tps& operator*=(double c);

      /** Divides by scalar and assigns. */
      Tps& operator/=(double c);

      /** Copy operator */
      Tps& operator =(const Tps& rhs);

      /** Adds the rhs series and assigns.   */
      Tps& operator+=(const Tps& rhs);

      /** Subtracts the rhs series and assigns. */
      Tps& operator-=(const Tps& rhs);

      /** Multiplies by the rhs series and assigns. */
      Tps& operator*=(const Tps& rhs);

      /** Divides by the rhs series and assigns. */
      Tps& operator/=(const Tps& rhs);

      // Additive & Multiplicative Operators
	
      /** Adds and returns the temporary object . */
      Tps operator+(const Tps& rhs) const;

      /** Subtracts and returns the temporary object. */
      Tps operator-(const Tps& rhs) const; 

      /** Multiplies and returns the temporary object. */
      Tps operator*(const Tps& rhs) const;

      /** Divides and returns the temporary object. */
      Tps operator/(const Tps& rhs) const;

      // Friends

      friend class VTps;

      // Dif. operators

      /** Returns the partial derivative of the zs series with respect to iv-variable.*/
      friend Tps D(const Tps& zs, unsigned int iv);

      /** Returns the Poisson bracket of zs1 and zs2 series:*/
      friend Tps poisson(const Tps& zs1, const Tps& zs2);

      // Other operators
      /** Unary minus.*/
      friend Tps operator-(const Tps& rhs);

      /** Returns the temporary object after addition of the rhs series and constant. */
      friend Tps operator+(const Tps& rhs, double c);

      /** Returns the temporary object after addition of constant and the rhs series.*/
      friend Tps operator+(double c, const Tps& rhs);

      /** Returns the temporary object after subtraction of the rhs series and constant. */
      friend Tps operator-(const Tps& rhs, double c);

      /** Returns the temporary object after subtraction of constant and the rhs series. */
      friend Tps operator-(double c, const Tps& rhs);

      /** Returns the temporary object after multiplication of  the rhs series and constant. */
      friend Tps operator*(const Tps& rhs, double c);

      /** Returns the temporary object after multiplication of constant and the rhs series. */
      friend Tps operator*(double c, const Tps& rhs);

      /** Returns the temporary object after division of the rhs series by constant. */
      friend Tps operator/(const Tps& rhs, double c);

      /** Returns the temporary object after division of  constant by the rhs series. */
      friend Tps operator/(double c, const Tps& rhs); 

      // Some useful math. functions

      /** Returns the square root of the zs series. */
      friend Tps sqrt(const Tps& zs);

    private:

      unsigned int _order;
      double* _tps;

      void initialize(double c, unsigned int order);
      void initialize(const Tps& rhs);
      void checkOrder(unsigned int order) const;
      void checkIndex(unsigned int index) const;
      void erase();
    };

}

// Attributes

inline unsigned int ZLIB::Tps::order() const { return _order; }
inline unsigned int ZLIB::Tps::size() const { return nmo(_order); }

// Access Operators

inline double& ZLIB::Tps::operator[](unsigned int index) 
{ 
  checkIndex(index); 
  return _tps[index]; 
}

inline double ZLIB::Tps::operator[](unsigned int index) const
{ 
  checkIndex(index); 
  return _tps[index]; 
}

#endif
        
