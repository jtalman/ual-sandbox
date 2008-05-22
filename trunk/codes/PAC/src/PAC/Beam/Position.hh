// Program     : PAC
// File        : PAC/Beam/Position.hh
// Copyright   : see Copyright file
// Description:  Position contains 6D canonical coordinates (x, px/p0, y, py/p0,
//                -c dt, de/p0c) defined with respect to the reference particle. 
// Author      : Nikolay Malitsky

#ifndef UAL_PAC_POSITION_HH
#define UAL_PAC_POSITION_HH

#include <vector>
#include "UAL/Common/AttributeSet.hh"

namespace PAC {

  /** Container of 6D canonical coordinates (x, px/p0, y, py/p0,
      -c*dt, de/p0c) defined with respect to the reference particle.
  */

  class Position  : public UAL::AttributeSet 
    {
    public:

      // Constructors &  destructor

      /** Constructor */
      Position();

      /** Copy constructor */
      Position(const Position& p);

      /** Destructor */
      virtual ~Position();

      /** Copy operator */
      const Position& operator =(const Position& p);

      // Access methods

      /** Returns the x-coordinate [m] */ 
      double  getX() const;

      /** Sets the x-coordinate [m] */
      void    setX(double v);

      /** Returns the  x-axis momentum of particle [rad] */
      double  getPX() const;

      /** Sets the  x-axis momentum of particle [rad] */
      void   setPX(double v);
  
      /** Returns the y-coordinate [m]. */
      double  getY() const;

      /** Sets the y-coordinate [m]. */
      void    setY(double v);
 
      /** Returns the y-axis momentum of particle [rad]. */
      double  getPY() const;

      /** Sets the y-axis momentum of particle [m]. */ 
      void  setPY(double v);

      /** Returns the ct-coordinate [m].*/
      double  getCT() const;

      /** Sets the ct-coordinate [m]. */
      void    setCT(double v);

      /** Returns the de-coordinate. */
      double  getDE() const;

      /** Sets the de-coordinate. */
      void  setDE(double v);

      /** Sets 6D canonical coordinates.*/
      void set(double x, double px, double y, double py,
	       double ct, double de);

      /** Returns the coordinate specified by index */
      double&  operator[](int index); 

      /** Returns the coordinate specified by index */
      double  operator[](int index) const; 

      /** Sets the i-th coordinate */
      void setCoordinate(int i, double v); 

      /** Returns the number of coordinates */  
      int size() const;

      // Assignment operators
  
      /** (Deprecated) Adds the Position object p to this object */
      const Position& operator+=(const Position& p);
  
      /** (Deprecated) Subtracts the Position object p from this object */
      const Position& operator-=(const Position& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v */
      const Position& operator*=(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value v */
      const Position& operator/=(double v);

      /** (Deprecated) Sums two Position objects . */
      Position  operator+(const Position& p);
  
      /** (Deprecated) Subtracts the Position object p from this object.*/
      Position  operator-(const Position& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      Position  operator*(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value. */
      Position  operator/(double v);

      /** (Deprecated) Subtracts the Position object p from this object. */
      // friend Position operator-(const Position& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      // friend Position operator*(double v, const Position& p);

    protected:
    
      /** Vector of coordinates */
      std::vector<double> m_data;
 
    };
}

#endif
        
