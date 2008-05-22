// Library     : PAC
// File        : Templates/PacMatrix.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_MATRIX_H
#define PAC_MATRIX_H

#include "PAC/Common/PacException.h"
#include "Templates/PacVector.h"


template <class T> class PacMatrix 
{
public:

  // Constructors

  PacMatrix(int rows = 0, int columns = 0)         {initialize(rows, columns);}
  PacMatrix(int rows, int columns, const T& value) {initialize(rows, columns, value);}
  PacMatrix(const PacMatrix<T>& matrix)            {initialize(matrix);}
  ~PacMatrix()                                     {erase();}

  // Assignment

  void operator = (const PacMatrix<T>& matrix)     {erase(); initialize(matrix);}

  // Access

  const PacVector<T>& operator[] (int i) const;
  PacVector<T>& operator[](int i);

  double  element(int i, int j) const              { return (*(_rows[i]))[j]; }
  double& element(int i, int j)                    { return (*(_rows[i]))[j]; }

  int rows() const                                 { return _rows.size();}
  int columns() const                              { return  rows() > 0 ? _rows[0]->size() : 0;}

protected:

  PacVector< PacVector<T>* > _rows;

  void initialize(int rows, int columns);
  void initialize(int rows, int columns, const T& value);
  void initialize(const PacMatrix& matrix);
  void erase();

};

template <class T> 
const PacVector<T>& PacMatrix<T>::operator[](int i) const
{
  if( i < 0 || i >= rows()){
    std::string msg = "Error : PacMatrix<T>::operator[](int i) : i is out of [0, rows] \n";
    PacDomainError(msg).raise();
  }
  return *(_rows[i]);
}

template <class T> 
PacVector<T>& PacMatrix<T>::operator[](int i)
{
  if( i < 0 || i >= rows()){
    std::string msg = "Error : PacMatrix<T>::operator[](int i) : i is out of [0, rows] \n";
    PacDomainError(msg).raise();
  }
  return *(_rows[i]);
}

template <class T> void PacMatrix<T>::initialize(int rows, int columns)
{
  _rows.reserve(rows);
  for(int i=0; i < rows; i++){
    _rows.push_back(new PacVector<T>(columns));
    if(!_rows[i]){
      std::string msg = "Error : PacMatrix<T>::initialize(int rows, int columns):";
             msg += "allocation failure \n";
      PacAllocError(msg).raise();
    }
  }
} 

template <class T> void PacMatrix<T>::initialize(int rows, int columns, const T& value)
{
  _rows.reserve(rows);
  for(int i=0; i < rows; i++){
    _rows.push_back(new PacVector<T>(columns, value));
    if(!_rows[i]){
      std::string msg  = "Error : PacMatrix<T>::initialize(int rows, int columns, const T& value) :";
             msg += "allocation failure \n";
      PacAllocError(msg).raise();
    }
  }
} 

template <class T> void PacMatrix<T>::initialize(const PacMatrix& matrix)
{
  initialize(matrix.rows(), matrix.columns());
  for(int i=0; i < matrix.rows(); i++) *(_rows[i]) = matrix[i];
} 

template <class T> void PacMatrix<T>::erase()
{
  for(unsigned int i=0; i < _rows.size(); i++) delete _rows[i];
  _rows.erase(_rows.begin(), _rows.end());
} 

#endif



  
