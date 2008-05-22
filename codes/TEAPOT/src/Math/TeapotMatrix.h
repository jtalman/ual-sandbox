// Library     : TEAPOT
// File        : Math/TeapotMatrix.h
// Copyright   : see Copyright file
// Author      : Steve Tepikian


/*

  Definitions for a matrix class.

*/

#if !defined(__TeapotMatrix_hxx__)
#define __TeapotMatrix_hxx__

#include "Math/TeapotVector.h"

class TeapotMatrix {
   // Useful operations on vectors.
   friend TeapotMatrix operator*(double val, const TeapotMatrix &vec);
   friend TeapotVector operator*(const TeapotMatrix &mat, const TeapotVector &val);

public:
   enum TeapotMatrixProperties { ORIG, LU_DECOMP };

   // Change of dimension. In reducing the dimensional size, memory
   // allocation is not reduced (use contract() if needed).
   void setDimension(int rows, int cols) { _reallocate(rows, cols); }

   // Vector dimension (length) and size (amount of allocated memory).
   int rows() const { return UM_numRows; }
   int columns() const;
   int size() const { return UM_size; }

   // The state of the internal matrix.
   TeapotMatrixProperties state() const { return UM_state; }

   // Matrix manipulation.

   double luDecomp();		// Returns the determinant.
   void luBksb(TeapotVector &b) const;
   TeapotMatrix inverse();

   // M = -S*Mt*S
   TeapotMatrix symplecticConjugation() const;

   void gaussJordan();
   void gaussJordan(TeapotVector &b);
   void gaussJordan(TeapotMatrix &b);

   // Contracts excess memory.
   void contract();

   // Access to data with range checks.
   TeapotVector&		operator[](int i);
   const TeapotVector&	operator[](int i) const;

   // Useful operations on vectors.
   TeapotMatrix&	operator=(const TeapotMatrix &vec);
   TeapotMatrix	operator+(const TeapotMatrix &vec) const;
   TeapotMatrix	operator-(const TeapotMatrix &vec) const;
   TeapotMatrix&	operator+=(const TeapotMatrix &vec);
   TeapotMatrix&	operator-=(const TeapotMatrix &vec);
   TeapotMatrix	operator*(const TeapotMatrix &val) const;
   TeapotMatrix	operator*(double val) const;

   TeapotMatrix() : UM_numRows(0), UM_size(0), UM_rows(NULL), UM_state(ORIG),
      UM_pivot(NULL) { }
   TeapotMatrix(int rows, int cols) { _allocate(rows, cols); }
   TeapotMatrix(const TeapotMatrix &mat);
   virtual ~TeapotMatrix();

private:
   enum UalMemoryStep { UMS_STEP = 8 };
   static TeapotVector	UM_nullVal;	// Used for out of range values.

   int			UM_numRows;	// The number of rows in matrix.
   int			UM_size;	// The number of rows allocated.
   TeapotVector*		UM_rows;	// The array of rows.

   TeapotMatrixProperties	UM_state;	// The state of the matrix stored.
   int*			UM_pivot;	// The index used for pivoting.

   // Handling memory allocation of the rows.
   void	_allocate(const int &rows, const int &cols);
   void	_deallocate();
   int	_newSize(const int &rows) const;

   // Does not contract allocated memory with a size reduction.
   void	_reallocate(const int &rows, const int &cols);

   TeapotMatrix(TeapotVector **array, const int &size, const int &dim);
};

inline int TeapotMatrix::_newSize(const int &rows) const {
   return UMS_STEP * ((rows / UMS_STEP) + 1);
}

inline TeapotMatrix::~TeapotMatrix() { _deallocate(); }

inline int TeapotMatrix::columns() const {
   return (UM_numRows > 0) ? UM_rows[0].dimension() : 0;
}

inline TeapotMatrix::TeapotMatrix(TeapotVector **array, const int &size,
			    const int &dim) :
   UM_numRows(dim),
   UM_size(size),
   UM_rows(*array),
   UM_state(ORIG)
{
   if (size > 0) {
      UM_pivot = new int [size];
   } else {
      UM_pivot = NULL;
   }
   *array = NULL;
}

#endif // TeapotMatrix.hxx
