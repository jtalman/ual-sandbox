// Library     : TEAPOT
// File        : Math/TeapotEigenBasis.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_EIGEN_BASIS_H
#define TEAPOT_EIGEN_BASIS_H

#include "Math/TeapotMatrix.h"

class TeapotEigenBasis
{
 public:

  // Constructors

  TeapotEigenBasis();
  TeapotEigenBasis(const TeapotEigenBasis& rhs); 
  TeapotEigenBasis(const TeapotMatrix& matrix);

  // Assignment
  
  TeapotEigenBasis& operator=(const TeapotEigenBasis& rhs);

  // Check the validity of this basis

  int isValid() const;

  // Return eigenvalues

  const TeapotVector& eigenValues() const;

  // Return eigenvectors

  const TeapotMatrix& eigenVectors() const;

 private:

  void initialize(int size);
  void define2D(const TeapotMatrix& matrix);
  
 private:

  int validity_;

  TeapotVector values_;
  TeapotMatrix vectors_;

};

#endif
