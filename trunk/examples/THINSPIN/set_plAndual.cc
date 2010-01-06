// Library       : THINSPIN
// File          : examples/THINSPIN/plAndul.cc
// Copyright     : see Copyright file
// Author        :
// C++ version   : J.Talman

  pl.set0( e0 + p0 * p.getDE());
  pl.set1(-p0 * p.getPX());
  pl.set2(-p0 * p.getPY());
  pl.set3(-sqrt(pl.get0()*pl.get0() - m0*m0 - pl.get1()*pl.get1() - pl.get2()*pl.get2()));

//std::cout << "Lab:  pl.get0() " << pl.get0() <<  " pl.get1()  " << pl.get1()  << " pl.get2()  " <<  pl.get2() << "  pl.get3() " <<  pl.get3() << "\n";
//THINSPIN::fourVector er;
//lorentzTransform(pl, betal, gamma, er);
//std::cout << "Rest: er.get0() " << er.get0() <<  " er.get1()  " << er.get1()  << " er.get2()  " <<  er.get2() << "  er.get3() " << er.get3() << "\n";
//THINSPIN::fourVector plM;
//lorentzTransform(er, betal*=(-1), gamma, plM);
//std::cout << "Lab: plM.get0() " << plM.get0() << " plM.get1() " << plM.get1() << " plM.get2() " << plM.get2() << "  plM.get3() " <<  plM.get3() << "\n\n";

  ul.set0(pl.get0()/m0);
  ul.set1(pl.get1()/m0);            //  negativeness due to covariant already included above
  ul.set2(pl.get2()/m0);
  ul.set3(pl.get3()/m0);

  gl = pl.get0()/m0;

