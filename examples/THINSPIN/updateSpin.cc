double teslaPerTw = p0 / e0 / data.m_l / rkicks;

THINSPIN::fourVector ulCon, slCon;

ulCon.set0( ulAve.get0());
ulCon.set1(-ulAve.get1());
ulCon.set2(-ulAve.get2());
ulCon.set3(-ulAve.get3());

slCon.set0( sl.get0());
slCon.set1(-sl.get1());
slCon.set2(-sl.get2());
slCon.set3(-sl.get3());

double B1tw=0,B2tw=0;
//getB1tw(mult, offset, rkicks, p[0], p[2], B1tw);
//getB2tw(mult, offset, rkicks, p[0], p[2], B2tw);
//B1tw=px;
//B2tw=py;

B1tw=Bxtw;
B2tw=Bytw;

double B1mks=0,B2mks=0;

B1mks = B1tw * teslaPerTw;
B2mks = B2tw * teslaPerTw;

double B1=B1mks,B2=B2mks,B3=0;
double E1=0    ,E2=0    ,E3=0;

Fl.comp00 = 0         ; Fl.comp01 = -E1         ; Fl.comp02 = -E2       ; Fl.comp03 = -E3;   // RF Cavity
Fl.comp10 = -Fl.comp01; Fl.comp11 = 0           ; Fl.comp12 = -B3       ; Fl.comp13 = B2 ;   // B2 - Dipole. B3 - Solenoid
Fl.comp20 = -Fl.comp02; Fl.comp21 = -Fl.comp12  ; Fl.comp22 = 0         ; Fl.comp23 = -B1;   // B1, B2 implicated in Multipoles as well
Fl.comp30 = -Fl.comp03; Fl.comp31 = -Fl.comp13  ; Fl.comp32 = -Fl.comp23; Fl.comp33 = 0  ;

/*
double f1 = CMB / M0 / UAL::clight;
double gover2minus1Fac = ba.getG() / UAL::clight / UAL::clight;;
double gover2          = ba.getG()+1;
*/

double f1 = 1;
double gover2minus1Fac = 1;
double gover2          = 1;

// MANIFESTLY FORM INVARIANT !!!
double Fl_times_sl0 = Fl.comp00*sl.get0() + Fl.comp01*sl.get1() + Fl.comp02*sl.get2() + Fl.comp03*sl.get3();
double Fl_times_sl1 = Fl.comp10*sl.get0() + Fl.comp11*sl.get1() + Fl.comp12*sl.get2() + Fl.comp13*sl.get3();
double Fl_times_sl2 = Fl.comp20*sl.get0() + Fl.comp21*sl.get1() + Fl.comp22*sl.get2() + Fl.comp23*sl.get3();
double Fl_times_sl3 = Fl.comp30*sl.get0() + Fl.comp31*sl.get1() + Fl.comp32*sl.get2() + Fl.comp33*sl.get3();

double Fl_times_ul0 = Fl.comp00*ul.get0() + Fl.comp01*ul.get1() + Fl.comp02*ul.get2() + Fl.comp03*ul.get3();
double Fl_times_ul1 = Fl.comp10*ul.get0() + Fl.comp11*ul.get1() + Fl.comp12*ul.get2() + Fl.comp13*ul.get3();
double Fl_times_ul2 = Fl.comp20*ul.get0() + Fl.comp21*ul.get1() + Fl.comp22*ul.get2() + Fl.comp23*ul.get3();
double Fl_times_ul3 = Fl.comp30*ul.get0() + Fl.comp31*ul.get1() + Fl.comp32*ul.get2() + Fl.comp33*ul.get3();

double sl_dot_Fl_times_ul = sl.get0()*Fl_times_ul0 + sl.get1()*Fl_times_ul1 +sl.get2()*Fl_times_ul2 +sl.get3()*Fl_times_ul3;

double gover2minus1FacFac = gover2minus1Fac * sl_dot_Fl_times_ul;

double dslCon0dTau = f1*(gover2*Fl_times_sl0 + gover2minus1FacFac*ulCon.get0()); 
double dslCon1dTau = f1*(gover2*Fl_times_sl1 + gover2minus1FacFac*ulCon.get1()); 
double dslCon2dTau = f1*(gover2*Fl_times_sl2 + gover2minus1FacFac*ulCon.get2()); 
double dslCon3dTau = f1*(gover2*Fl_times_sl3 + gover2minus1FacFac*ulCon.get3()); 

double dslCon0dt = dslCon0dTau / gl;        //  gl before last kick?
double dslCon1dt = dslCon1dTau / gl; 
double dslCon2dt = dslCon2dTau / gl; 
double dslCon3dt = dslCon3dTau / gl; 

THINSPIN::threeVector v;
v.setX(-ulAve.get1()/gl/m0);
v.setY(-ulAve.get2()/gl/m0);
v.setZ(-ulAve.get3()/gl/m0);

double vs = v.getZ();     // ???
//std::cout << "vs = " << vs << "(m / s)\n";

double dslCon0ds = dslCon0dt / vs; 
double dslCon1ds = dslCon1dt / vs; 
double dslCon2ds = dslCon2dt / vs; 
double dslCon3ds = dslCon3dt / vs; 
