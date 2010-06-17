double deltap0byp0=0;
double deltapxbyp0=-Bytw;
double deltapybyp0= Bxtw;
double deltapsbyp0= Bytw*plAve.get1()/plAve.get3()-Bxtw*plAve.get2()/plAve.get3();

/*
double pAve=sqrt(plAve.get0()*plAve.get0() - m0*m0);
deltapsbyp0=sqrt(pAve*pAve/p0/p0-(pAve.get1()/p0-Bytw)*(pAve.get1()/p0-Bytw)-(pAve.get2()/p0+Bxtw)*(pAve.get2()/p0+Bxtw);
deltapsbyp0-=pAve.get3()/p0;
*/

       fac         = G0*p0/m0/m0;
double fac2        = sl.get1()*deltapxbyp0+sl.get2()*deltapybyp0+sl.get3()*deltapsbyp0;
       deltaslCon0 = plAve.get0()*fac*fac2;
       deltaslCon1 = plAve.get1()*fac*fac2;
       deltaslCon2 = plAve.get2()*fac*fac2;
       deltaslCon3 = plAve.get3()*fac*fac2;

//THINSPIN::fourVector deltaSL;

       deltaSLsecond.set0( deltaslCon0);
       deltaSLsecond.set1(-deltaslCon1);
       deltaSLsecond.set2(-deltaslCon2);
       deltaSLsecond.set3(-deltaslCon3);

//     SL[ip]+=deltaSL;
