// Program     :
// File        : 
// Copyright   : see Copyright file
// Description :  
// Author      :

#ifndef THINSPIN_FOUR_VECTOR_HH
#define THINSPIN_FOUR_VECTOR_HH

#include <iostream>
#include <vector>

namespace THINSPIN {
  class fourVector
    {
    public:

      // Constructors &  destructor

      /** Constructor */
      fourVector();

      /** Copy constructor */
      fourVector(const fourVector& p);

      /** Destructor */
      virtual ~fourVector();

      /** Copy operator */
      const fourVector& operator =(const fourVector& p);

      // Access methods

      double  get0() const;

      void    set0(double v);

      double  get1() const;

      void   set1(double v);
  
      double  get2() const;

      void    set2(double v);
 
      double  get3() const;

      void  set3(double v);

      /** Sets 4D canonical coordinates.*/
      void set(double v0, double v1, double v2, double v3);

      /** Returns the coordinate specified by index */
      double&  operator[](int index); 

      /** Returns the coordinate specified by index */
      double  operator[](int index) const; 

      /** Sets the i-th coordinate */
      void setCoordinate(int i, double v); 

      /** Returns the number of coordinates */  
      int size() const;

      void print(){
//       std::cout << "THINSPIN::fourVector " << name << "  ";
         std::cout << "get0() = " << get0() << ", get1() = " << get1() << ", get2() = " << get2() << ", get3() = " << get3() << "\n";
      }

      // Assignment operators
  
      /** (Deprecated) Adds the fourVector object p to this object */
      const fourVector& operator+=(const fourVector& p);
  
      /** (Deprecated) Subtracts the fourVector object p from this object */
      const fourVector& operator-=(const fourVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v */
      const fourVector& operator*=(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value v */
      const fourVector& operator/=(double v);

      /** (Deprecated) Sums two fourVector objects . */
      fourVector  operator+(const fourVector& p);
  
      /** (Deprecated) Subtracts the fourVector object p from this object.*/
      fourVector  operator-(const fourVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      fourVector  operator*(double v);
  
      /** (Deprecated) Divides all coordinates of this object by the value. */
      fourVector  operator/(double v);

      /** (Deprecated) Subtracts the fourVector object p from this object. */
      // friend fourVector operator-(const fourVector& p);
  
      /** (Deprecated) Multiplies all coordinates of this object by the value v. */
      // friend fourVector operator*(double v, const fourVector& p);

    protected:
    
      /** Vector of coordinates */
      std::vector<double> m_data;
 
    };
}

#endif
        
