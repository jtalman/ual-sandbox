#define bunchSize 1

double m0;                                     // design     mass                GeV
double p0;                                     // design lab central momentum    GeV
double e0;                                     // design lab central energy      GeV
double c0;                                     // design     charge              C
double G0;                                     // design     gyromagnetic anomaly/ratio  dimensionless ca .001

PAC::Spin spin(1,0,0);                         // Sx, Sy, Sz are laboratory frame contravariant components of the spin
                                               // This is different from SPINK in which Sx, Sy, Sz stand for particle rest frame components

THINSPIN::fourVector sl;                       // (implicitly covariant) spin lab  frame 4 vector
THINSPIN::fourVector sr;                       // (implicitly covariant) spin rest frame 4 vector has sr0 = 0

THINSPIN::fourVector SL[bunchSize];            // (implicitly covariant) spin lab  frame 4 vector
THINSPIN::fourVector SR[bunchSize];            // (implicitly covariant) spin rest frame 4 vector has sr0 = 0

//THINSPIN::fourVector slCon;                    // lab contravariant 4 spin vector

                                               //              note that current means before spin kick/update
THINSPIN::fourVector ul;                       // (implicitly covariant) current lab 4 velocity (reused by each bunch particle in its propagate loop)
THINSPIN::fourVector pl;                       // (implicitly covariant) current lab 4 momentum (reused by each bunch particle in its propagate loop)
double gl;                                     // current lab gamma

THINSPIN::fourTensor Fl("Fl");                 // (implicitly contravariant) rank 2 lab field tensor
THINSPIN::fourTensor FLtw("FLtw");             // (implicitly contravariant) rank 2 "integrated" lab field tensor

//double delta = .01;
  double delta = 1;
//double delta = .05;
double sl0 = 0;                                // set in tracker
double sl1 = 0;                                // set in tracker
double sl2 = 0;                                // set in tracker
double sl3 = 0;                                // set in tracker

double tolerance = .01;
double sr0 =  0;
double sr1 =  0;
double sr2 =  0;
double sr3 = -1;

double SL0[bunchSize];
double SL1[bunchSize];
double SL2[bunchSize];
double SL3[bunchSize];

double SR0[bunchSize];
double SR1[bunchSize];
double SR2[bunchSize];
double SR3[bunchSize];
