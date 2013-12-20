#include<stdlib.h>
#include<iostream>
#include<iomanip>
#include<cmath>

#include"matrix.h"
 
using namespace std;
 
 matrix::matrix()//default constructor
 {
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    a[i][j]=0.;
   }
  }
 }
 
 void matrix::set()// to set matrix elements
 {
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    cerr<<"\n Enter "<<i<<","<<j<<" element=";
    cin>>a[i][j];
   }
  }
 }

 void matrix::show()// to show matrix elements
 {
  cerr<<"\n Matrix is=\n";
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    cerr<< setw(10) << setfill('0') << a[i][j]<<",";
   }
   cerr<<"\n";
  }
 }

 void matrix::show(string header)// to show matrix elements
 {
  cerr<< header << " is\n";
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    cerr<< setw(10) << setfill('0') << a[i][j]<<",";
   }
   cerr<<"\n";
  }
 }

 matrix::matrix matrix::operator*(matrix X)
 {
  matrix C;
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    for(int k=0;k<3;k++)
    {
     C.a[i][j]=C.a[i][j]+a[i][k]*X.a[k][j];       
    }
   }
  }
  return(C);
 }

 void matrix::setRoll(double phi){
  a[0][0]=+cos(phi);
  a[0][1]=-sin(phi);
  a[0][2]=+0.;
  a[1][0]=+sin(phi);
  a[1][1]=+cos(phi);
  a[1][2]=+0.;
  a[2][0]=+0.;
  a[2][1]=+0.;
  a[2][2]=+1.;
 }

 void matrix::setYaw(double yAngle){
  a[0][0]=+cos(yAngle);
  a[0][1]=+0.;
  a[0][2]=-sin(yAngle);
  a[1][0]=+0.;
  a[1][1]=+1.;
  a[1][2]=+0.;
  a[2][0]=+sin(yAngle);
  a[2][1]=+0.;
  a[2][2]=+cos(yAngle);
 }

 void matrix::setIdentity(){
  a[0][0]=+1.;
  a[0][1]=+0.;
  a[0][2]=+0.;
  a[1][0]=+0.;
  a[1][1]=+1.;
  a[1][2]=+0.;
  a[2][0]=+0.;
  a[2][1]=+0.;
  a[2][2]=+1.;
 }

 matrix matrix::deltaFromId(){
  matrix C;
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
    C.a[i][j]=a[i][j];
   }
  }

  C.a[0][0]-=+1.;
  C.a[0][1]-=+0.;
  C.a[0][2]-=+0.;
  C.a[1][0]-=+0.;
  C.a[1][1]-=+1.;
  C.a[1][2]-=+0.;
  C.a[2][0]-=+0.;
  C.a[2][1]-=+0.;
  C.a[2][2]-=+1.;

  return C;
 }
