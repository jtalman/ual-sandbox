// Library     : TEAPOT
// File        : Math/TeapotVector.h
// Copyright   : see Copyright file
// Author      : Steve Tepikian

/*

  A vector class definition.

*/

#ifndef __TeapotVector_hxx__
#define __TeapotVector_hxx__


#include <iostream>
#include <string>
#include <math.h>
#include <string.h>

class TeapotMatrix;

using namespace std;

class TeapotVector {
   // Useful operations on vectors.
   friend TeapotVector operator*(double val, const TeapotVector &vec);
   friend double dot(const TeapotVector &v1, const TeapotVector &v2);
   friend TeapotVector operator*(const TeapotMatrix &mat, const TeapotVector &val);

public:
   // Change of dimension. In reducing the dimensional size, memory
   // allocation is not reduced (use contract() if needed).
   void setDimension(int dim) { _reallocate(dim); }

   // Vector dimension (length) and size (amount of allocated memory).
   int dimension() const { return UV_dim; }
   int size() const { return UV_size; }

   // Contracts excess memory.
   void contract();

   // Access to data with range checks.
   double &		operator[](int i);
   const double &	operator[](int i) const;

   // Useful operations on vectors.
   TeapotVector&	operator=(const TeapotVector &vec);
   TeapotVector	operator+(const TeapotVector &vec) const;
   TeapotVector	operator-(const TeapotVector &vec) const;
   TeapotVector&	operator+=(const TeapotVector &vec);
   TeapotVector&	operator-=(const TeapotVector &vec);
   TeapotVector	operator*(double val) const;

   // Useful functions.
   void zero();		// Sets all elements to zero.
   void unit(int i);	// Sets the i'th element to one and others to zero.

   // Constructors and destructors.
   TeapotVector() : UV_dim(0), UV_size(0), UV_array(NULL) { }
   TeapotVector(int dim);
   TeapotVector(const TeapotVector &vec);
   virtual ~TeapotVector();

private:
   enum UalMemoryStep { UMS_STEP = 8 };
   static double	UV_nullVal;	// Used when accessing out of range.

   int		UV_dim;		// The dimension of the vector.
   int		UV_size;	// The amount of memory allocated.
   double*	UV_array;	// The pointer to the allocated memory.

   // Handling memory allocation of the array.
   void	_allocate(const int &dim);
   void	_deallocate();
   int	_newSize(const int &dim) const;

   // Does not contract allocated memory with a size reduction.
   void	_reallocate(const int &dim);

   TeapotVector(double **array, const int &size, const int &dim);
};

inline int TeapotVector::_newSize(const int &dim) const {
   return UMS_STEP * ((dim / UMS_STEP) + 1);
}

inline void TeapotVector::_allocate(const int &dim) {
   UV_array = new double [UV_size = _newSize(UV_dim = dim)];
}

inline void TeapotVector::_deallocate() {
   delete [] UV_array;
   UV_array = NULL;
   UV_dim = UV_size = 0;
}

inline TeapotVector::TeapotVector(int dim) { _allocate(dim); }
inline TeapotVector::~TeapotVector() { _deallocate(); }

inline TeapotVector::TeapotVector(const TeapotVector &vec) {
   _allocate(vec.UV_dim);
   memcpy(UV_array, vec.UV_array, UV_dim * sizeof(double));
}

inline TeapotVector::TeapotVector(double **array, const int &size, const int &dim) :
   UV_dim(dim),
   UV_size(size),
   UV_array(*array)
{
   *array = NULL;
}

inline TeapotVector& TeapotVector::operator=(const TeapotVector &vec)
{
   if (this != &vec) {
      _reallocate(vec.UV_dim);
      memcpy(UV_array, vec.UV_array, UV_dim * sizeof(double));
   }
   return *this;
}

inline void TeapotVector::zero()
{
   memset(UV_array, 0, UV_size * sizeof(double));
}

inline void TeapotVector::unit(int i)
{
   memset(UV_array, 0, UV_size * sizeof(double));
   if (0 <= i && i < UV_dim) UV_array[i] = 1.0;
}


#endif // Ends TeapotVector.hxx
