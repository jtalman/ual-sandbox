double clight     = UAL::clight;                   // m/s  
double GeVperJ    = 1/UAL::elemCharge/1e9;         //
double JperGeV    = 1/GeVperJ/clight/clight;       //

double pmass      = UAL::pmass;                    // proton rest mass
double m0         = pmass;                         //
double m0MKS      = pmass*JperGeV;
double massMKS    = pmass*JperGeV;
double chge       = UAL::elemCharge;               // C 1.6e-19   ; // proton charge

double E0         = 17e6;                          // V/m
double gap        = 2e-2;                          // m
