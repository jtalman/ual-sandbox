// Library     : TEAPOT
// File        : Math/TeapotMatrix.cc
// Copyright   : see Copyright file
// Author      : Steve Tepikian

/*

  The member functions for the TeapotMatrix class.

*/

#include "Math/TeapotMatrix.h"


// --------------- Simple functions -----------------------------

inline void SWAP(double &a, double &b) { double c = a; a = b; b = c; }


// --------------- Static data members --------------------------

TeapotVector TeapotMatrix::UM_nullVal;


// --------------- Member functions -----------------------------

// ##### Constructors.
TeapotMatrix::TeapotMatrix(const TeapotMatrix &mat)
{
   if (mat.UM_numRows == 0) {
      UM_size = UM_numRows = 0;
      UM_rows = NULL;
      UM_pivot = NULL;
      UM_state = ORIG;
   } else {   
      _allocate(mat.UM_numRows, mat.UM_rows[0].dimension());
      for (int i = 0; i < UM_numRows; i++) UM_rows[i] = mat.UM_rows[i];
   }
}

// ##### Subscript operators.
TeapotVector& TeapotMatrix::operator[](int i)
{
   if (0 <= i && i < UM_numRows) {
      return UM_rows[i];
   } else {
      cerr << "Warning -- Out of range for subscript operator in TeapotMatrix."
	   << " File: " << __FILE__ << " at line: " << __LINE__ << endl;
      return UM_nullVal;
   }
}

const TeapotVector& TeapotMatrix::operator[](int i) const
{
   if (0 <= i && i < UM_numRows) {
      return UM_rows[i];
   } else {
      cerr << "Warning -- Out of range for subscript operator in TeapotMatrix."
	   << " File: " << __FILE__ << " at line: " << __LINE__ << endl;
      return UM_nullVal;
   }
}

// ##### Memory allocation functions.
void TeapotMatrix::contract()
{
   if (UM_size > _newSize(UM_numRows)) {
      TeapotVector *temp = new TeapotVector [UM_size = _newSize(UM_numRows)];
      int *tpivot = new int [UM_size];
      for (int i = 0; i < UM_numRows; i++) {
	 tpivot[i] = UM_pivot[i];
	 temp[i] = UM_rows[i];
      }
      delete [] UM_pivot;
      delete [] UM_rows;
      UM_rows = temp;
      UM_pivot = tpivot;
   }
}

void TeapotMatrix::_allocate(const int &rows, const int &cols) {
   UM_rows = new TeapotVector [UM_size = _newSize(UM_numRows = rows)];
   UM_pivot = new int [UM_size];
   for (int i = 0; i < UM_numRows; i++) {
      UM_rows[i].setDimension(cols);
      UM_pivot[i] = i;
   }
   UM_state = ORIG;
}

void TeapotMatrix::_reallocate(const int &rows, const int &cols)
{
   // Check to see if there is enough allocated memory.
   if (UM_size == 0) {
      _allocate(rows, cols);
   } else {
      if (rows > UM_size) {
	 int oldRows = UM_numRows, *pivot = UM_pivot;
	 TeapotVector *temp = UM_rows;
	 UM_rows = new TeapotVector [UM_size = _newSize(UM_numRows = rows)];
	 UM_pivot = new int [UM_size];
	 for (int i = 0; i < UM_numRows; i++) {
	    if (i < oldRows) {
	       UM_rows[i] = temp[i];
	       UM_pivot[i] = pivot[i];
	    } else {
	       UM_pivot[i] = i;
	    }
	    UM_rows[i].setDimension(cols);
	 }
	 delete [] pivot;
	 delete [] temp;
      }

      // Setting the new row size.
      UM_state = ORIG;
   }
}

void TeapotMatrix::_deallocate()
{
   delete [] UM_pivot;
   delete [] UM_rows;
   UM_rows = NULL;
   UM_pivot = NULL;
   UM_numRows = UM_size = 0;
   UM_state = ORIG;
}

// ##### Useful operations with matrices
TeapotMatrix operator*(double val, const TeapotMatrix &vec)
{
   TeapotVector *temp = new TeapotVector [vec.UM_size];
   for (int i = 0; i < vec.UM_numRows; i++) temp[i] = val * vec.UM_rows[i];
   return TeapotMatrix(&temp, vec.UM_size, vec.UM_numRows);
}

TeapotMatrix& TeapotMatrix::operator=(const TeapotMatrix &mat)
{
   if (this != &mat) {
      if (mat.UM_numRows == 0) {
	 _deallocate();
      } else {   
	 _reallocate(mat.UM_numRows, mat.UM_rows[0].dimension());
	 for (int i = 0; i < UM_numRows; i++) UM_rows[i] = mat.UM_rows[i];
      }
   }
   return *this;
}

TeapotMatrix TeapotMatrix::operator+(const TeapotMatrix &vec) const
{
   if (vec.UM_numRows != UM_numRows) {
      cerr << "Warning -- Incompatible dimensions in TeapotMatrix. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int size, dim = (vec.UM_numRows < UM_numRows) ? vec.UM_numRows : UM_numRows;
   TeapotVector *temp = new TeapotVector [size = _newSize(dim)];
   for (int i = 0; i < dim; i++) temp[i] = UM_rows[i] + vec.UM_rows[i];
   return TeapotMatrix(&temp, size, dim);
}

TeapotMatrix TeapotMatrix::operator-(const TeapotMatrix &vec) const
{
   if (vec.UM_numRows != UM_numRows) {
      cerr << "Warning -- Incompatible dimensions in TeapotMatrix. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int size, dim = (vec.UM_numRows < UM_numRows) ? vec.UM_numRows : UM_numRows;
   TeapotVector *temp = new TeapotVector [size = _newSize(dim)];
   for (int i = 0; i < dim; i++) temp[i] = UM_rows[i] - vec.UM_rows[i];
   return TeapotMatrix(&temp, size, dim);
}

TeapotMatrix& TeapotMatrix::operator+=(const TeapotMatrix &vec)
{
   if (vec.UM_numRows != UM_numRows) {
      cerr << "Warning -- Incompatible dimensions in TeapotMatrix. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int dim = (vec.UM_numRows < UM_numRows) ? vec.UM_numRows : UM_numRows;
   for (int i = 0; i < dim; i++) UM_rows[i] += vec.UM_rows[i];
   return *this;
}

TeapotMatrix& TeapotMatrix::operator-=(const TeapotMatrix &vec)
{
   if (vec.UM_numRows != UM_numRows) {
      cerr << "Warning -- Incompatible dimensions in TeapotMatrix. File: "
	   << __FILE__ << " at line: " << __LINE__ << endl;
   }
   int dim = (vec.UM_numRows < UM_numRows) ? vec.UM_numRows : UM_numRows;
   for (int i = 0; i < dim; i++) UM_rows[i] += vec.UM_rows[i];
   return *this;
}

TeapotVector operator*(const TeapotMatrix &mat, const TeapotVector &val)
{
   int dim = mat.UM_numRows, size = val._newSize(dim);
   double  *ans = new double [size];

   if (mat.UM_state == TeapotMatrix::LU_DECOMP) {
      int i, ii = -1, ip, j;
      double sum;

      for (i = 0; i < dim; i++) ans[i] = val[i];
      for (i = 0; i < dim; i++) {
	 ip = mat.UM_pivot[i];
	 sum = ans[ip];
	 ans[ip] = ans[i];
	 if (ii != -1) {
	    for (j = ii; j <= i - 1; j++) sum -= mat.UM_rows[i][j] * ans[j];
	 } else if (sum) {
	    ii = i;
	 }
	 ans[i] = sum;
      }
      for (i = dim - 1; i >= 0; i--) {
	 sum = ans[i];
	 for (j = i + 1; j < dim; j++) sum -= mat.UM_rows[i][j] * ans[j];
	 ans[i] = sum / mat.UM_rows[i][i];
      }
   } else {
      for (int i = 0; i < dim; i++) ans[i] = dot(mat.UM_rows[i], val);
   }
   return TeapotVector(&ans, size, dim);
}

TeapotMatrix TeapotMatrix::operator*(const TeapotMatrix &mat) const
{
   if (UM_state == LU_DECOMP || mat.UM_state == LU_DECOMP) {
      cerr << "Error -- Trying multiply a LU Decomposed TeapotMatrix. File: "
	   << __FILE__ << " line: " << __LINE__ << endl;
      return TeapotMatrix(NULL, 0, 0);
   }
   if (UM_numRows == 0 || mat.UM_numRows == 0) {
      cerr << "Error -- Trying multiply a NULL TeapotMatrix. File: "
	   << __FILE__ << " line: " << __LINE__ << endl;
      return TeapotMatrix(NULL, 0, 0);
   }
   if (UM_rows[0].dimension() != mat.UM_numRows) {
      cerr << "Error -- Incompatible dimensions for multiplying UalMatrices."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return TeapotMatrix(NULL, 0, 0);
   }

   int i, j, k, dim = UM_numRows, size = _newSize(dim);
   double sum;
   TeapotVector *ans = new TeapotVector [size];

   for (i = 0; i < UM_numRows; i++) {
      ans[i].setDimension(mat.columns());
      for (j = 0; j < mat.columns(); j++) {
	 for (sum = 0.0, k = 0; k < mat.UM_numRows; k++) {
	    sum += UM_rows[i][k] * mat.UM_rows[k][j];
	 }
	 ans[i][j] = sum;
      }
   }
   return TeapotMatrix(&ans, size, dim);
}

TeapotMatrix TeapotMatrix::operator*(double val) const
{
   TeapotVector *temp = new TeapotVector [UM_size];
   for (int i = 0; i < UM_numRows; i++) temp[i] = val * UM_rows[i];
   return TeapotMatrix(&temp, UM_size, UM_numRows);
}

void TeapotMatrix::luBksb(TeapotVector &b) const
{
   if (UM_numRows == 0 || UM_state != LU_DECOMP) return;

   int i, ii = -1, ip, j;
   double sum;

   for (i = 0; i < UM_numRows; i++) {
      ip = UM_pivot[i];
      sum = b[ip];
      b[ip] = b[i];
      if (ii != -1) {
	 for (j = ii; j <= i - 1; j++) sum -= UM_rows[i][j] * b[j];
      } else if (sum) {
	 ii = i;
      }
      b[i] = sum;
   }
   for (i = UM_numRows - 1; i >= 0; i--) {
      sum = b[i];
      for (j = i + 1; j < UM_numRows; j++) sum -= UM_rows[i][j] * b[j];
      b[i] = sum / UM_rows[i][i];
   }
}

double TeapotMatrix::luDecomp()
{
   if (UM_state == LU_DECOMP) return 0.0;
   if (UM_numRows == 0) {
      cerr << "Error -- Trying LU decomposition on a NULL TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return 0.0;
   }
   if (UM_numRows != UM_rows[0].dimension()) {
      cerr << "Error -- Trying LU decomposition on a non-square TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return 0.0;
   }

   int i, imax, j, k;
   double det, big, dum, sum, temp;

   TeapotVector vv(UM_numRows);
   const double TINY = 1.e-20;

   UM_state = LU_DECOMP;
   det = 1.0;
   for (i = 0; i < UM_numRows; i++) {
      big = 0.0;
      for (j = 0; j < UM_numRows; j++) {
         if ((temp = fabs(UM_rows[i][j])) > big) big = temp;
      }
      if (big == 0.0) {
         cerr << "Error -- A singular TeapotMatrix. File: " << __FILE__
	      << " at line: " << __LINE__ << endl;
         return 0.0;
      }
      vv[i] = 1.0 / big;
   }
   for (j = 0; j < UM_numRows; j++) {
 
      for (i = 0; i < j; i++) 
	{
         sum = UM_rows[i][j];
         for (k = 0; k < i; k++) sum -= UM_rows[i][k] * UM_rows[k][j];
         UM_rows[i][j] = sum;

      }

      big = 0.0;
      imax = 0;                   
      for (i = j; i < UM_numRows; i++) {
         sum = UM_rows[i][j];
	 for (k = 0; k < j; k++) sum -= UM_rows[i][k] * UM_rows[k][j];
         UM_rows[i][j] = sum;

         if ((dum = vv[i] * fabs(sum)) >= big) {
            big = dum;
            imax = i;

         }

      }

      if (j != imax) {
         for (k = 0; k < UM_numRows; k++) {
            dum = UM_rows[imax][k];
            UM_rows[imax][k] = UM_rows[j][k];
            UM_rows[j][k] = dum;
         }
         det = -det;
         vv[imax] = vv[j];
      }
      UM_pivot[j] = imax;
      if (UM_rows[j][j] == 0.0) UM_rows[j][j] = TINY;
      if (j != UM_numRows - 1) {
         dum = 1.0 / (UM_rows[j][j]);
         for (i = j + 1; i < UM_numRows; i++) UM_rows[i][j] *= dum;
      }
   }

   for (i = 0; i < UM_numRows; i++) { det *= UM_rows[i][i];}

   return det;
}
/*
TeapotMatrix TeapotMatrix::inverse()
{
  double detr = luDecomp();
  
  if(detr == 0) {
      cerr << "Error -- Trying Inverse method on a matrix with Determinant == 0"
	   << " TeapotMatrix. File: " << __FILE__ << " line: " << __LINE__
	   << endl;
      return TeapotMatrix(0, 0);
   }

  TeapotVector *temp = new TeapotVector [UM_size];
  TeapotVector col(UM_numRows), zero(UM_numRows);

  for(int i = 0; i < UM_numRows; i++) temp[i].setDimension(UM_numRows);
  for(int j = 0; j < UM_numRows; j++) {
    col    = zero;
    col[j] = 1.;
    luBksb(col);
    for(int i = 0; i < UM_numRows; i++) temp[i][j] = col[i];
  }
  return TeapotMatrix(&temp, UM_size, UM_numRows);
}
*/

TeapotMatrix TeapotMatrix::inverse()
{
  TeapotMatrix temp(*this);

  double detr = temp.luDecomp();
  int i, j;

  TeapotMatrix y(UM_numRows, UM_numRows);
  
  if(detr == 0) {
      cerr << "Error -- Trying Inverse method on a matrix with Determinant == 0"
	   << " TeapotMatrix. File: " << __FILE__ << " line: " << __LINE__
	   << endl;
      return y;
  }

  TeapotVector col(UM_numRows), zero(UM_numRows);

  for(j = 0; j < UM_numRows; j++) zero[j] = 0.0;
  for(j = 0; j < UM_numRows; j++) {
    col    = zero;
    col[j] = 1.;
    temp.luBksb(col);
    for(i = 0; i < UM_numRows; i++) y[i][j] = col[i];
  }
  return y;
}


void TeapotMatrix::gaussJordan()
{
   if (UM_state == LU_DECOMP) {
      cerr << "Error -- Trying Gauss-Jordan method on a LU Decomposed"
	   << " TeapotMatrix. File: " << __FILE__ << " line: " << __LINE__
	   << endl;
      return;
   }
   if (UM_numRows == 0) {
      cerr << "Error -- Trying Gauss-Jordan method on a NULL TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }
   if (UM_numRows != UM_rows[0].dimension()) {
      cerr << "Error -- Trying Gauss-Jordan method on a non-square TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }

   int *indxc, *indxr;
   int i, icol, irow, j, k, l, ll;
   double big, dum, pivinv;

   indxc = new int [UM_numRows];
   indxr = new int [UM_numRows];
   for (j = 0; j < UM_numRows; j++) UM_pivot[j] = -1;
   for (i = 0; i < UM_numRows; i++) {
      big = 0.0;
      icol = 0;      
      irow = 0;     
      for (j = 0; j < UM_numRows; j++) {
	 if (UM_pivot[j] != 0)
	    for (k = 0; k < UM_numRows; k++) {
	       if (UM_pivot[k] == -1) {
		  if (fabs(UM_rows[j][k]) >= big) {
		     big = fabs(UM_rows[j][k]);
		     irow = j;
		     icol = k;
		  }
	       } else if (UM_pivot[k] > 0) {
		  cerr << "Warning -- GAUSSJ: Singular Matrix-1" << endl;
	       }
	    }
      }
      ++(UM_pivot[icol]);
      if (irow != icol) {
	 for (l = 0; l < UM_numRows; l++) {
	    SWAP(UM_rows[irow][l], UM_rows[icol][l]);
	 }
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if (UM_rows[icol][icol] == 0.0) {
	 cerr << "Warning -- GAUSSJ: Singular Matrix-2" << endl;
      }
      pivinv = 1.0 / UM_rows[icol][icol];
      UM_rows[icol][icol] = 1.0;
      for (l = 0; l < UM_numRows; l++) UM_rows[icol][l] *= pivinv;
      for (ll = 0; ll < UM_numRows; ll++) {
	 if (ll != icol) {
	    dum = UM_rows[ll][icol];
	    UM_rows[ll][icol] = 0.0;
	    for (l = 0; l < UM_numRows; l++) {
	       UM_rows[ll][l] -= UM_rows[icol][l] * dum;
	    }
	 }
      }
   }
   for (l = UM_numRows - 1; l >= 0; l--) {
      if (indxr[l] != indxc[l]) {
	 for (k = 0; k < UM_numRows; k++) {
	    SWAP(UM_rows[k][indxr[l]], UM_rows[k][indxc[l]]);
	 }
      }
   }
   delete [] indxr;
   delete [] indxc;
}

void TeapotMatrix::gaussJordan(TeapotVector &b)
{
   if (UM_state == LU_DECOMP) {
      cerr << "Error -- Trying Gauss-Jordan method on a LU Decomposed"
	   << " TeapotMatrix. File: " << __FILE__ << " line: " << __LINE__
	   << endl;
      return;
   }
   if (UM_numRows == 0) {
      cerr << "Error -- Trying Gauss-Jordan method on a NULL TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }
   if (UM_numRows != UM_rows[0].dimension()) {
      cerr << "Error -- Trying Gauss-Jordan method on a non-square TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }

   int *indxc, *indxr;
   int i, icol, irow, j, k, l, ll;
   double big, dum, pivinv;

   indxc = new int [UM_numRows];
   indxr = new int [UM_numRows];
   for (j = 0; j < UM_numRows; j++) UM_pivot[j] = -1;
   for (i = 0; i < UM_numRows; i++) {
      big = 0.0;
      icol = 0;  
      irow = 0;  
      for (j = 0; j < UM_numRows; j++) {
	 if (UM_pivot[j] != 0)
	    for (k = 0; k < UM_numRows; k++) {
	       if (UM_pivot[k] == -1) {
		  if (fabs(UM_rows[j][k]) >= big) {
		     big = fabs(UM_rows[j][k]);
		     irow = j;
		     icol = k;
		  }
	       } else if (UM_pivot[k] > 0) {
		  cerr << "Warning -- GAUSSJ: Singular Matrix-1" << endl;
	       }
	    }
      }
      ++(UM_pivot[icol]);
      if (irow != icol) {
	 for (l = 0; l < UM_numRows; l++) {
	    SWAP(UM_rows[irow][l], UM_rows[icol][l]);
	 }
	 SWAP(b[irow], b[icol]);
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if (UM_rows[icol][icol] == 0.0) {
	 cerr << "Warning -- GAUSSJ: Singular Matrix-2" << endl;
      }
      pivinv = 1.0 / UM_rows[icol][icol];
      UM_rows[icol][icol] = 1.0;
      for (l = 0; l < UM_numRows; l++) UM_rows[icol][l] *= pivinv;
      b[icol] *= pivinv;
      for (ll = 0; ll < UM_numRows; ll++) {
	 if (ll != icol) {
	    dum = UM_rows[ll][icol];
	    UM_rows[ll][icol] = 0.0;
	    for (l = 0; l < UM_numRows; l++) {
	       UM_rows[ll][l] -= UM_rows[icol][l] * dum;
	    }
	    b[ll] -= b[icol] * dum;
	 }
      }
   }
   for (l = UM_numRows - 1; l >= 0; l--) {
      if (indxr[l] != indxc[l]) {
	 for (k = 0; k < UM_numRows; k++) {
	    SWAP(UM_rows[k][indxr[l]], UM_rows[k][indxc[l]]);
	 }
      }
   }
   delete [] indxr;
   delete [] indxc;
}

void TeapotMatrix::gaussJordan(TeapotMatrix &b)
{
   if (UM_state == LU_DECOMP) {
      cerr << "Error -- Trying Gauss-Jordan method on a LU Decomposed"
	   << " TeapotMatrix. File: " << __FILE__ << " line: " << __LINE__
	   << endl;
      return;
   }
   if (UM_numRows == 0) {
      cerr << "Error -- Trying Gauss-Jordan method on a NULL TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }
   if (UM_numRows != UM_rows[0].dimension()) {
      cerr << "Error -- Trying Gauss-Jordan method on a non-square TeapotMatrix."
	   << " File: " << __FILE__ << " line: " << __LINE__ << endl;
      return;
   }

   int *indxc, *indxr;
   int i, icol, irow, j, k, l, ll;
   double big, dum, pivinv;

   indxc = new int [UM_numRows];
   indxr = new int [UM_numRows];
   for (j = 0; j < UM_numRows; j++) UM_pivot[j] = -1;
   for (i = 0; i < UM_numRows; i++) {
      big = 0.0;
      icol = 0;  
      irow = 0;  
      for (j = 0; j < UM_numRows; j++) {
	 if (UM_pivot[j] != 0)
	    for (k = 0; k < UM_numRows; k++) {
	       if (UM_pivot[k] == -1) {
		  if (fabs(UM_rows[j][k]) >= big) {
		     big = fabs(UM_rows[j][k]);
		     irow = j;
		     icol = k;
		  }
	       } else if (UM_pivot[k] > 0) {
		  cerr << "Warning -- GAUSSJ: Singular Matrix-1" << endl;
	       }
	    }
      }
      ++(UM_pivot[icol]);
      if (irow != icol) {
	 for (l = 0; l < UM_numRows; l++) {
	    SWAP(UM_rows[irow][l], UM_rows[icol][l]);
	 }
	 for (l = 0; l < b.columns(); l++) SWAP(b[irow][l], b[icol][l]);
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if (UM_rows[icol][icol] == 0.0) {
	 cerr << "Warning -- GAUSSJ: Singular Matrix-2" << endl;
      }
      pivinv = 1.0 / UM_rows[icol][icol];
      UM_rows[icol][icol] = 1.0;
      for (l = 0; l < UM_numRows; l++) UM_rows[icol][l] *= pivinv;
      for (l = 0; l < b.columns(); l++) b[icol][l] *= pivinv;
      for (ll = 0; ll < UM_numRows; ll++) {
	 if (ll != icol) {
	    dum = UM_rows[ll][icol];
	    UM_rows[ll][icol] = 0.0;
	    for (l = 0; l < UM_numRows; l++) {
	       UM_rows[ll][l] -= UM_rows[icol][l] * dum;
	    }
	    for (l = 0; l < b.columns(); l++) b[ll][l] -= b[icol][l] * dum;
	 }
      }
   }
   for (l = UM_numRows - 1; l >= 0; l--) {
      if (indxr[l] != indxc[l]) {
	 for (k = 0; k < UM_numRows; k++) {
	    SWAP(UM_rows[k][indxr[l]], UM_rows[k][indxc[l]]);
	 }
      }
   }
   delete [] indxr;
   delete [] indxc;
}

// Symplectic conjugation: M = -S*Mt*S
TeapotMatrix TeapotMatrix::symplecticConjugation() const
{
  int i, j;

  int size = rows();
  if(size != columns()) {
    cerr << "Warning -- Incompatible dimensions in TeapotMatrix. " << endl
	 << "rows(" << size << ") != columns (" << columns() << ")"<< endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;
    return *this;
  } 

  // Transpose and negate matrix
  TeapotMatrix Mt(size, size);

  for(i = 0; i < size; i++)
    for(j = 0; j < size; j++){
      Mt[i][j] = -UM_rows[j][i];
    } 

  // Make S
  if((size/2)*2 != size) {
    cerr << "Warning -- Incompatible dimensions in TeapotMatrix. " << endl
	 << "size(" << size << ") is not even " << endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;
    return *this;
  } 

  TeapotMatrix S(size, size);

  for(i = 0; i < size; i++)
    for(j = 0; j < size; j++)
      S[i][j] = 0.0;

  for(i = 0; i < size; i += 2){
    S[i][i+1] = -1.;
    S[i+1][i] = +1.;
  }

  return S*Mt*S;
}
