THINSPIN::fourVector sl;        // for legibility of formulae

sl = SL[ip];

double E1tw=Extw;
double E2tw=Eytw;
double E3tw=Estw;
double B1tw=Bxtw;
double B2tw=Bytw;
double B3tw=Bstw;

FLtw.comp00 = 0           ; FLtw.comp01 = -E1tw       ; FLtw.comp02 = -E2tw       ; FLtw.comp03 = -E3tw;    // RF Cavity
FLtw.comp10 = -FLtw.comp01; FLtw.comp11 = 0           ; FLtw.comp12 = -B3tw       ; FLtw.comp13 = B2tw;   // B2 - Dipole. B3 - Solenoid
FLtw.comp20 = -FLtw.comp02; FLtw.comp21 = -FLtw.comp12; FLtw.comp22 = 0           ; FLtw.comp23 = -B1tw;// B1, B2 implicated in Multipoles as well
FLtw.comp30 = -FLtw.comp03; FLtw.comp31 = -FLtw.comp13; FLtw.comp32 = -FLtw.comp23; FLtw.comp33 = 0;

double fac = p0 * (G0+1) / pl.get3();

// MANIFESTLY FORM INVARIANT !!!
double FLtw_times_sl0 = FLtw.comp00*sl.get0() + FLtw.comp01*sl.get1() + FLtw.comp02*sl.get2() + FLtw.comp03*sl.get3();
double FLtw_times_sl1 = FLtw.comp10*sl.get0() + FLtw.comp11*sl.get1() + FLtw.comp12*sl.get2() + FLtw.comp13*sl.get3();
double FLtw_times_sl2 = FLtw.comp20*sl.get0() + FLtw.comp21*sl.get1() + FLtw.comp22*sl.get2() + FLtw.comp23*sl.get3();
double FLtw_times_sl3 = FLtw.comp30*sl.get0() + FLtw.comp31*sl.get1() + FLtw.comp32*sl.get2() + FLtw.comp33*sl.get3();

double deltaslCon0 = fac*FLtw_times_sl0; 
double deltaslCon1 = fac*FLtw_times_sl1; 
double deltaslCon2 = fac*FLtw_times_sl2; 
double deltaslCon3 = fac*FLtw_times_sl3; 

//THINSPIN::fourVector deltaSL;
deltaSLfirst.set0( deltaslCon0);
deltaSLfirst.set1(-deltaslCon1);
deltaSLfirst.set2(-deltaslCon2);
deltaSLfirst.set3(-deltaslCon3);

//SL[ip]+=deltaSL;
