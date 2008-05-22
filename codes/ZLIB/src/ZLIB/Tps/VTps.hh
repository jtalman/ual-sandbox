// Library     : ZLIB
// File        : ZLIB/Tps/VTps.hh
// Description : Vector of Truncated Power Series (TPS)
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky & Alexander Reshetov

#ifndef UAL_ZLIB_VTPS_HH
#define UAL_ZLIB_VTPS_HH

#include "ZLIB/Tps/Tps.hh"
#include "ZLIB/Tps/Vector.hh"

namespace ZLIB {

      /** Returns the partial derivative of all Tps components with respect to iv-variable.*/
      VTps D(const VTps& zs, int iv);  

      /** Returns the Poisson bracket of the TPS (zs1) and VTPS (zs2) */                 
      VTps poisson(const Tps& zs1, const VTps& zs2);

      /** Returns the Poisson bracket of the VTPS (zs1)  and TPS (zs2) */   
      VTps poisson(const VTps& zs1, const Tps& zs2);   

      // Other friend operators

      /** Unary minus.*/
      VTps operator-(const VTps& zs);

      /** Returns the temporary object after addition of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      VTps operator+(const VTps& zs, double c);

      /** Returns the temporary object after addition of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      VTps operator+(double c, const VTps& zs);

      /** Returns the temporary object after subtraction of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      VTps operator-(const VTps& zs, double c);

      /** Returns the temporary object after subtraction of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      VTps operator-(double c, const VTps& zs);

      /** Returns the temporary object after multiplication of the VTPS (zs) and constant. */ 
      VTps operator*(const VTps& zs, double c);

      /** Returns the temporary object after multiplication of the VTPS (zs) and constant. */ 
      VTps operator*(double c, const VTps& zs);

      /** Returns the temporary object after division of the VTPS (zs) by constant. */ 
      VTps operator/(const VTps& zs, double c);

      /** Returns the temporary object after division of constant by the VTPS (zs). */ 
      VTps operator/(double c, const VTps& zs); 

      /** Writes Tps components into the output stream. */
      ostream& operator<<(ostream& out, const VTps& vtps);

      /** Reads Tps components from the input stream. */
      istream& operator>>(istream& in,  VTps& vtps);

  /** Represents the data and algebra of Vectors of Truncated Power Series (VTPS). */

  class VTps : public ZLIB::Space
    {
    public:

      /** Constructor */
      explicit VTps(unsigned int size = 0);

      /** Copy constructor */
      VTps(const VTps& rhs);

      /** Destructor */
      virtual ~VTps();

      // Attributes 

      /** Returns the order of Tps components.*/
      unsigned int order() const;

      /** Sets the order for all Tps components. */
      void order(unsigned int o);

      /** Returns the number of Tps components. */
      unsigned int size() const;

      /** Sets the number of Tps components. */
      void size(unsigned int s);

      /** Returns the jth coefficient of the ith TPS component */
      double  vtps(unsigned int i, unsigned int j) const;

      /** Returns a reference to the jth coefficient of the ith TPS component */
      double& vtps(unsigned int i, unsigned int j);

      // Operators

      // Access operators

      /** Returns a reference to the jth Tps component. */
      Tps& operator[](unsigned int j);

      /** Returns a constant reference to the jth Tps component. */
      const Tps& operator[](unsigned int index) const;

      // Assignment operators

      /** Converts and assigns */
      VTps& operator =(double c);

      /** Adds the identity matrix multiplied by constant and assigns. */
      VTps& operator+=(double c);

      /** Subtracts the identity matrix multiplied by constant and assigns. */
      VTps& operator-=(double c);

      /** Multiplies by constant and assigns. */ 
      VTps& operator*=(double c);

      /** Divides by scalar and assigns. */
      VTps& operator/=(double c);  

      /** Copy operator */
      VTps& operator =(const VTps& rhs);

      /** Adds the VTPS object and assigns */ 
      VTps& operator+=(const VTps& rhs);

      /** Subtracts the VTPS object and assigns. */
      VTps& operator-=(const VTps& rhs);

      /** Multiplies by the VTPS object and assigns. */
      VTps& operator*=(const VTps& rhs);

      /** Divides by the VTPS object and assigns. */
      VTps& operator/=(const VTps& rhs);

      // Additive & Multiplicative Operators

      /** Adds and returns the temporary object . */
      VTps operator+(const VTps& rhs);

      /** Subtracts and returns the temporary object. */
      VTps operator-(const VTps& rhs); 

      /** Multiplies and returns the temporary object. */
      VTps operator*(const VTps& rhs);

      /** Divides and returns the temporary object. */
      VTps operator/(const VTps& rhs); 

      // Mapping

      /** Propagates the vector of doubles */
      void propagate (Vector& v) const;

      // Dif. operators

      /** Returns the partial derivative of all Tps components with respect to iv-variable.*/
      friend VTps D(const VTps& zs, int iv);  

      /** Returns the Poisson bracket of the TPS (zs1) and VTPS (zs2) */                 
      friend VTps poisson(const Tps& zs1, const VTps& zs2);

      /** Returns the Poisson bracket of the VTPS (zs1)  and TPS (zs2) */   
      friend VTps poisson(const VTps& zs1, const Tps& zs2);   

      // Other friend operators

      /** Unary minus.*/
      friend VTps operator-(const VTps& zs);

      /** Returns the temporary object after addition of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      friend VTps operator+(const VTps& zs, double c);

      /** Returns the temporary object after addition of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      friend VTps operator+(double c, const VTps& zs);

      /** Returns the temporary object after subtraction of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      friend VTps operator-(const VTps& zs, double c);

      /** Returns the temporary object after subtraction of the VTPS (zs) and the identity 
	  matrix multiplied by constant (c). */
      friend VTps operator-(double c, const VTps& zs);

      /** Returns the temporary object after multiplication of the VTPS (zs) and constant. */ 
      friend VTps operator*(const VTps& zs, double c);

      /** Returns the temporary object after multiplication of the VTPS (zs) and constant. */ 
      friend VTps operator*(double c, const VTps& zs);

      /** Returns the temporary object after division of the VTPS (zs) by constant. */ 
      friend VTps operator/(const VTps& zs, double c);

      /** Returns the temporary object after division of constant by the VTPS (zs). */ 
      friend VTps operator/(double c, const VTps& zs); 

      // I/0

      /** Reads Tps components from the file with the specified name. */
      void read(const char* file);

      /** Writes Tps components into the file with the specified name. */
      void write(const char* file);
 

      /** Writes Tps components into the output stream. */
      friend ostream& operator<<(ostream& out, const VTps& vtps);

      /** Reads Tps components from the input stream. */
      friend istream& operator>>(istream& in,  VTps& vtps);

    private:

      unsigned int _size;
      Tps* _vtps;

      void initialize(unsigned int s);
      void initialize(const VTps& zvs);
      void checkIndex(unsigned int index) const ; 
      void compareSize(unsigned int s) const ;
      void erase();

    };

}

// Vector size

inline unsigned int ZLIB::VTps::size() const { return _size; }

// Fast access to TPS coefficients

inline double  ZLIB::VTps::vtps(unsigned int i, unsigned int j) const { return _vtps[i]._tps[j]; }
inline double& ZLIB::VTps::vtps(unsigned int i, unsigned int j)       { return _vtps[i]._tps[j]; }

// Access operators

inline ZLIB::Tps& ZLIB::VTps::operator[] (unsigned int index)
{
  checkIndex(index);
  return _vtps[index];
}

inline const ZLIB::Tps& ZLIB::VTps::operator[] (unsigned int index) const
{
  checkIndex(index);
  return _vtps[index];
}

#endif
