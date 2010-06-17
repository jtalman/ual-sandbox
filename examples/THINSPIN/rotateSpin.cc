double t, sx,sy,sz;

//sl=SL[ip];
 
sx=SL[ip].get1();
sy=SL[ip].get2();
sz=SL[ip].get3();
//   sl[0] = sl[0];  
t = ((slice.cphpl())*sx);
t -= ((slice.sphpl())*sz);         // sx*cos(phi+) - sz*sin(phi+)
SL[ip].set1(t);
//   sl[2] = sl[2];  
t = ((slice.cphpl())*sz);
t += ((slice.sphpl())*sx);         // sz*cos(phi+) + sx*sin(phi+)
//sl[3] = t;
SL[ip].set3(t);

std:cout << "rotateSpin SL[" << ip << "].get0() " << SL[ip].get0() << " get1() " << SL[ip].get1() << " get2() " << SL[ip].get2() << " get3() " << SL[ip].get3() << "\n"; 

//if(fabs(sl.get0()-sl0)<delta && fabs(sl.get1()-sl1)<delta && fabs(sl.get2()-sl2)<delta && fabs(sl.get3()-sl3)<delta){
// sl.print();
//}
  
/*
if(fabs(sr.get0()-sr0)<tolerance && fabs(sr.get1()-sr1)<tolerance && fabs(sr.get2()-sr2)<tolerance && fabs(sr.get3()-sr3)<tolerance){
   sr.print();
}
*/
//sr.print();
