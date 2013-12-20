#include<stdlib.h>
#include<iostream>
#include<iomanip>
#include<cmath>
 
class matrix
{
 public:
  matrix();
  void set();
  void show();
  void show(std::string header);
  void setRoll(double phi);
  void setYaw (double yAngle);
  void setIdentity();
  matrix deltaFromId();
  matrix operator*(matrix x);

  double a[3][3];
};
