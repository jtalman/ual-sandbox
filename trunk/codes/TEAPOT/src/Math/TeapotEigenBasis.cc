// Library     : TEAPOT
// File        : Math/TeapoEigenBasis.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include <stdlib.h>
#include <math.h>
#include "Math/TeapotEigenBasis.h"


// Constructor
TeapotEigenBasis::TeapotEigenBasis()
  : validity_(0), values_(0), vectors_(0, 0)
{
}

// Constructor
TeapotEigenBasis::TeapotEigenBasis(const TeapotEigenBasis& rhs)
  : validity_(rhs.validity_), values_(rhs.values_), vectors_(rhs.vectors_)
{
}

// Constructor
TeapotEigenBasis::TeapotEigenBasis(const TeapotMatrix& matrix)
  : validity_(0)
{
  int rows = matrix.rows();
  if(rows != matrix.columns()) {
    cerr << "Warning -- Incompatible dimensions in TeapotMatrix. " << endl
	 << "rows(" << rows << ") != columns (" << matrix.columns() << ")"<< endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;
    exit(1);
  } 
  
  switch(rows) {
  case 4: 
    define2D(matrix);
    break;
  default:
    cerr << "Error -- Incompatible dimensions in TeapotMatrix. " << endl
	 << "rows(" << rows << ") != 4 or 6"<< endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;
    exit(1);
  };

}

// Copy operator
TeapotEigenBasis& TeapotEigenBasis::operator=(const TeapotEigenBasis& rhs)
{
  values_   = rhs.values_;
  vectors_  = rhs.vectors_;
  validity_ = rhs.validity_;
  return *this;
}

// Check the validity of this basis
int TeapotEigenBasis::isValid() const
{
  return validity_;
}

// Return eigenvalues
const TeapotVector& TeapotEigenBasis::eigenValues() const
{
  return values_;
}

// Return eigenvectors
const TeapotMatrix& TeapotEigenBasis::eigenVectors() const
{
  return vectors_;
}

// Define eigenvectors from the 2D transfer matrix
void TeapotEigenBasis::define2D(const TeapotMatrix& matrix)
{
  initialize(matrix.rows());

  int i, j, size = 2;
  TeapotMatrix A(size, size), B(size, size), C(size, size), D(size, size);

  // Partition the transfer matrix

  for(i=0; i < size; i++){
    for(j=0; j < size; j++){   
     A[i][j] = matrix[i][j];
     B[i][j] = matrix[i][j + size];
     C[i][j] = matrix[i + size][j];
     D[i][j] = matrix[i + size][j + size];
    }
  }

  double trA = A[0][0] + A[1][1];
  double trD = D[0][0] + D[1][1];

  if(trA == 0 || trD == 0 ) {
    cerr << "Warning trA(" << trA << ") or trD(" << trD << ") == 0" << endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;
    return;
  }

  // E = B + C_bar

  TeapotMatrix E(size, size), Ebar(size, size); 

  // Temporary implementation of E matrix. It should be based on Matrix
  // methods, such as transpose(), S(), and simplectic conjugation. 

  E[0][0] = C[0][0] + B[1][1]; 
  E[0][1] = C[0][1] - B[0][1];
  E[1][0] = C[1][0] - B[1][0];  
  E[1][1] = C[1][1] + B[0][0]; 

  Ebar[0][0] =  E[1][1]; 
  Ebar[0][1] = -E[0][1];
  Ebar[1][0] = -E[1][0];  
  Ebar[1][1] =  E[0][0]; 

  double detE = E[0][0]*E[1][1] - E[0][1]*E[1][0];

  // Find eigenvalues

  double trA_D = trA - trD;
  int signA_D = (trA_D > 0.0) ? 1 : -1;

  values_[0] = (trA + trD + signA_D*sqrt(trA_D*trA_D + 4.0*detE))/2.0;
  values_[1] = (trA + trD - signA_D*sqrt(trA_D*trA_D + 4.0*detE))/2.0;  

  // Find eigenvectors

  TeapotMatrix I(size, size); 

  I = I*0.0;
  for(i=0; i < size; i++) I[i][i] = 1.0;

  TeapotMatrix RA(size, size), RD(size, size);

  double vA_trD = values_[0] - trD;
  double vD_trA = values_[1] - trA;

  if(vA_trD == 0.0 || vD_trA == 0.0) {
    cerr << "Warning: eigenvalue A - trD (" << vA_trD << ") or "
	 << " eigenvalue D - trA (" << vD_trA << ") == 0" << endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;   
    return;
  }
  
  RA = E *(1./(values_[0] - trD));
  RD = Ebar*(1./(values_[1] - trA));

  if(values_[0] == values_[1]){
    cerr << "Warning: eigenvalues A == B == " << values_[0] << endl
	 << "File: " << __FILE__ << " at line: " << __LINE__ << endl;   
    return;
  }

  double gfac = sqrt(fabs(vD_trA/(values_[0] - values_[1])));
  
  for(i=0; i < size; i++){
    for(j=0; j < size; j++){   
     vectors_[i][j]               =  gfac*I[j][i];
     vectors_[i][j + size]        =  gfac*RA[j][i];
     vectors_[i + size][j]        =  gfac*RD[j][i];
     vectors_[i + size][j + size] =  gfac*I[j][i];
    }
  }

  validity_ = 1;
}


// Initialize eigenvalues by 0.0 and eigenvectors by I matrix
void TeapotEigenBasis::initialize(int size)
{
  values_.setDimension(size);
  vectors_.setDimension(size, size); 

  for(int i = 0; i < size; i++){
    values_[i] = 0.0;
    for(int j = 0; j < size; j++){ 
      vectors_[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
}





