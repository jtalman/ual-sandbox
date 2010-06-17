// Program     : 
// File        : 
// Copyright   : see Copyright file
// Description : 
// Author      : J.Talman

#ifndef THINSPIN_THREE_VECTOR_HH
#define THINSPIN_THREE_VECTOR_HH

#include <string>
#include <iostream>
#include <ctime> 
#include <vector> 

namespace THINSPIN {
   class threeVector {
    public:
      // Constructors &  destructor

      /** Constructor */
      threeVector();

      /** Copy constructor */
      threeVector(const threeVector& p);

      /** Destructor */
      virtual ~threeVector();

      /** Copy operator */
      const threeVector& operator =(const threeVector& p);

      // Access methods

      /** Returns the x-coordinate [m] */ 
      double  getX() const;

      /** Sets the x-coordinate [m] */
      void    setX(double v);

      /** Returns the y-coordinate [m]. */
      double  getY() const;

      /** Sets the y-coordinate [m]. */
      void    setY(double v);
 
      /** Returns the y-axis momentum of particle [rad]. */
      double  getZ() const;

      /** Sets the y-axis momentum of particle [m]. */ 
      void  setZ(double v);

      /** Sets 3D canonical coordinates.*/
      void set(double x, double y, double z);

      /** Returns the coordinate specified by index */
      double&  operator[](int index); 

      /** Returns the coordinate specified by index */
      double  operator[](int index) const; 

      /** Sets the i-th coordinate */
      void setCoordinate(int i, double v); 

      /** Returns the number of coordinates */  
      int size() const;

      // Assignment operators
  
      /** (Deprecated) Adds the threeVector object p to this object */
      const threeVector& operator+=(const threeVector& p);
  
      /** (Deprecated) Subtracts the threeVector object p from this object */
      const threeVector& operator-=(const threeVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v */
      const threeVector& operator*=(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value v */
      const threeVector& operator/=(double v);

      /** (Deprecated) Sums two threeVector objects . */
      threeVector  operator+(const threeVector& p);
  
      /** (Deprecated) Subtracts the threeVector object p from this object.*/
      threeVector  operator-(const threeVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      threeVector  operator*(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value. */
      threeVector  operator/(double v);

      /** (Deprecated) Subtracts the threeVector object p from this object. */
      // friend threeVector operator-(const threeVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      // friend threeVector operator*(double v, const threeVector& p);

    protected:
    
      /** Vector of coordinates */
      std::vector<double> m_data;
 
    };
}

#endif
