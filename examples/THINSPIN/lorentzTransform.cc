#ifndef THINSPIN_LORENTZ_TRANSFORM_CC
#define THINSPIN_LORENTZ_TRANSFORM_CC

void lorentzTransform(const THINSPIN::fourVector& in, const THINSPIN::threeVector beta, const double gamma, THINSPIN::fourVector& out){
   double magBetaSqu = beta.getX()*beta.getX() + beta.getY()*beta.getY() + beta.getZ()*beta.getZ();
   double fac = (gamma-1)/magBetaSqu;
   double dot = beta.getX()*in.get1() + beta.getY()*in.get2() + beta.getZ()*in.get3();

   out.set0(gamma*(in.get0()+dot));

   out.set1(in.get1()+beta.getX()*(fac*dot + gamma*in.get0()));
   out.set2(in.get2()+beta.getY()*(fac*dot + gamma*in.get0()));
   out.set3(in.get3()+beta.getZ()*(fac*dot + gamma*in.get0()));
}

#endif
