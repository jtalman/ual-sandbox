// Library     : ZLIB
// File        : ZLIB/Tps/Vector.h
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky 


#ifndef UAL_ZLIB_VECTOR_HH
#define UAL_ZLIB_VECTOR_HH

#include "ZLIB/Tps/Def.hh"

namespace ZLIB {


/** Vector of double elements propagated by the VTPS object. */

  class Vector
    {
    public:

      // Constructors & destructor

      /** Constructor */
      Vector(unsigned int s = 0, double value = 0.0);

      /** Copy constructor */
      Vector(const Vector& rhs);

      /** Destructor */
      virtual ~Vector();

      // Copy operator

      /** Copy operator */
      Vector& operator = (const Vector& rhs);

      // Access operators

      /** Returns the ith element. */
      double  operator[](unsigned int i) const;

      /** Returns a reference to the ith element */
      double& operator[](unsigned int i);

      // Vector size

      /** Returns the vector size */
      unsigned int size();

    private:

      // Data

      double* vector_;
      unsigned int size_;

      void initialize(const Vector& rhs);
      void erase();

    };
}

// Vector size

inline unsigned int ZLIB::Vector::size() { return size_; }

// Access operators

inline double  ZLIB::Vector::operator[](unsigned int index) const { return vector_[index]; }
inline double& ZLIB::Vector::operator[](unsigned int index)       { return vector_[index]; }

#endif
