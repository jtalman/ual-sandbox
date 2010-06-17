#include "threeVector.hh"
#include "fourVector.hh"
#include "fourTensor.hh"
#include "lorentzTransform.cc"
#include "lorentzTransformCon.cc"

extern double m0;                                     // design lab mass                GeV
extern double p0;                                     // design lab central momentum    GeV
extern double e0;                                     // design lab central energy      GeV
extern double c0;
extern double G0;

extern PAC::Spin spin;

extern THINSPIN::fourVector sl;                         // spin lab  frame 4 vector
extern THINSPIN::fourVector sr;                         // spin rest frame 4 vector has sr0 = 0

extern THINSPIN::fourVector SL[];            // (implicitly covariant) spin lab  frame 4 vector
extern THINSPIN::fourVector SR[];            // (implicitly covariant) spin rest frame 4 vector has sr0 = 0 

                                               //              note that current means before spin kick/update
extern THINSPIN::fourVector ul;                // (implicitly covariant) current lab 4 velocity (reused by each bunch particle in its propagate loop)
extern THINSPIN::fourVector pl;                // (implicitly covariant) current lab 4 momentum (reused by each bunch particle in its propagate loop)
extern double gl;

extern THINSPIN::fourVector ul;                       // (implicitly covariant) lab 4 velocity
extern THINSPIN::fourVector el;                       // (implicitly covariant) lab 4 momentum
extern THINSPIN::fourTensor Fl;                       // (implicitly contravariant) rank 2 lab field tensor
extern THINSPIN::fourTensor FLtw;                     // (implicitly contravariant) rank 2 "integrated" lab field tensor

extern double delta;
extern double SL0[];
extern double SL1[];
extern double SL2[];
extern double SL3[];

extern double tolerance;
extern double SR0[];
extern double SR1[];
extern double SR2[];
extern double SR3[];
