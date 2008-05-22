// Library     : TEAPOT
// File        : Math/TeapotVector.cc
// Copyright   : see Copyright file
// Author      : Steve Tepikian

/*

  Member functions for the TeapotVector class.

*/


#include "Math/TeapotVector.h"


// --------------- Static data members --------------------------

double TeapotVector::UV_nullVal = 0.0;


// --------------- Member functions -----------------------------

// ##### Subscript operators.
double & TeapotVector::operator[](int i)
{
   if (0 <= i && i < UV_dim) {
      return UV_array[i];
   } else {
      cerr << "Warning -- Out of range for subscript operator in TeapotVector."
	   << " File: " << __FILE__ << " at line: " << __LINE__ << endl;
      return UV_nullVal;
   }
}

const double & TeapotVector::operator[](int i) const
{
   if (0 <= i && i < UV_dim) {
      return UV_array[i];
   } else {
      cerr << "Warning -- Out of range for subscript operator in TeapotVector."
	   << " File: " << __FILE__ << " at line: " << __LINE__ << endl;
      return UV_nullVal;
   }
}

// ##### Memory allocation functions.
void TeapotVector::contract()
{
   if (UV_size > _newSize(UV_dim)) {
      double *temp;
      temp = new double [UV_size = _newSize(UV_dim)];
      memcpy(temp, UV_array, UV_dim * sizeof(double));
      delete [] UV_array;
      UV_array = temp;
   }
}

void TeapotVector::_reallocate(const int &dim)
{
   // Check to see if there is enough allocated memory.
   if (UV_size == 0) {
      UV_array = new double [UV_size = _newSize(UV_dim = dim)];
   } else {
      if (dim > UV_size) {
	 int oldDim = UV_dim;
	 double *temp = UV_array;
	 UV_array = new double [UV_size = _newSize(UV_dim = dim)];
	 if (oldDim > 0) memcpy(UV_array, temp, oldDim * sizeof(double));
	 delete [] temp;
      }
   }
}

// ##### Useful operations with vectors
TeapotVector operator*(double val, const TeapotVector &vec)
{
   double *temp = new double [vec.UV_size];
   for (int i = 0; i < vec.UV_dim; i++) temp[i] = val * vec.UV_array[i];
   return TeapotVector(&temp, vec.UV_size, vec.UV_dim);
}

TeapotVector TeapotVector::operator+(const TeapotVector &vec) const
{
   if (vec.UV_dim != UV_dim) {
      cerr << "Warning -- Incompatible dimensions in TeapotVector. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int size, dim = (vec.UV_dim < UV_dim) ? vec.UV_dim : UV_dim;
   double *temp = new double [size = _newSize(dim)];
   for (int i = 0; i < dim; i++) temp[i] = UV_array[i] + vec.UV_array[i];
   return TeapotVector(&temp, size, dim);
}

TeapotVector TeapotVector::operator-(const TeapotVector &vec) const
{
   if (vec.UV_dim != UV_dim) {
      cerr << "Warning -- Incompatible dimensions in TeapotVector. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int size, dim = (vec.UV_dim < UV_dim) ? vec.UV_dim : UV_dim;
   double *temp = new double [size = _newSize(dim)];
   for (int i = 0; i < dim; i++) temp[i] = UV_array[i] - vec.UV_array[i];
   return TeapotVector(&temp, size, dim);
}

TeapotVector& TeapotVector::operator+=(const TeapotVector &vec)
{
   if (vec.UV_dim != UV_dim) {
      cerr << "Warning -- Incompatible dimensions in TeapotVector. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int dim = (vec.UV_dim < UV_dim) ? vec.UV_dim : UV_dim;
   for (int i = 0; i < dim; i++) UV_array[i] += vec.UV_array[i];
   return *this;
}

TeapotVector& TeapotVector::operator-=(const TeapotVector &vec)
{
   if (vec.UV_dim != UV_dim) {
      cerr << "Warning -- Incompatible dimensions in TeapotVector. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int dim = (vec.UV_dim < UV_dim) ? vec.UV_dim : UV_dim;
   for (int i = 0; i < dim; i++) UV_array[i] -= vec.UV_array[i];
   return *this;
}

TeapotVector TeapotVector::operator*(double val) const
{
   double *temp = new double [UV_size];
   for (int i = 0; i < UV_dim; i++) temp[i] = val * UV_array[i];
   return TeapotVector(&temp, UV_size, UV_dim);
}

double dot(const TeapotVector &v1, const TeapotVector &v2)
{
   double ans = 0.0;
   if (v1.UV_dim != v2.UV_dim) {
      cerr << "Warning -- Incompatible dimensions in TeapotVector. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int dim = (v1.UV_dim < v2.UV_dim) ? v1.UV_dim : v2.UV_dim;
   for (int i = 0; i < dim; i++) ans += v1.UV_array[i] * v2.UV_array[i];
   return ans;
}
