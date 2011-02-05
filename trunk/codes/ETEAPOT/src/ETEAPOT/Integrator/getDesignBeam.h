double e0=cba.getEnergy();
double m0=cba.getMass();
double q0=cba.getCharge();
double t0=cba.getElapsedTime();
double f0=cba.getRevfreq();
double M0=cba.getMacrosize();
double G0=cba.getG();
double L0=cba.getL();
double E0=cba.getE();
double R0=cba.getR();

double GeVperJ   = 1./q0/1.e9;                            // units
double p0        = sqrt(e0*e0-m0*m0);                     // derived beam momentum
double g0        = e0/m0;                                 // derived beam gamma
double b0        = sqrt(1-1/g0/g0);                       // derived beam beta (same as velocity)

double m         = -1.2;                                  // exact field index