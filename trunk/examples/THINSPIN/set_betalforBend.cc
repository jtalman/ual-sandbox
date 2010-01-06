// Library       : THINSPIN
// File          : examples/THINSPIN/set_betalForBend.cc
// Copyright     : see Copyright file
// Author        :
// C++ version   : J.Talman

double e = e0 + pos.getDE() * p0;
double pp = sqrt(e*e - m0*m0);
double gamma = e / m0;
double pX = p0 * pos.getPX();                                   // gamma M0 vx
double pY = p0 * pos.getPY();                                   // gamma M0 vy
double pZ = sqrt(pp*pp - pX*pX - pY*pY);                      // gamma M0 vz

THINSPIN::threeVector betal;

betal.setX(pX / gamma / m0);
betal.setY(pY / gamma / m0);
betal.setZ(pZ / gamma / m0);

double beta = sqrt(betal.getX()*betal.getX() + betal.getY()*betal.getY() + betal.getZ()*betal.getZ());

