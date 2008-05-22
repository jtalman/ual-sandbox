//A linkDef file is needed because of the namespace.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// This line is also default for this class.
// the + tells rootcint to make the streamer.

//  these lines deal with the namespace
#pragma link C++ namespace UAL;
#pragma link C++ namespace PAC;
#pragma link C++ namespace ZLIB;
#pragma link C++ nestedclasses;

#pragma link C++ class PAC::BeamAttributes-!;
#pragma link C++ class PAC::Position-!;
#pragma link C++ class PAC::Bunch-!;
#pragma link C++ class PAC::Particle-!;
#pragma link C++ class PAC::Spin-!;
#pragma link C++ class PacVTps-!;
#pragma link C++ class PacTMap-!;
#pragma link C++ class PacTwissData-!;
#pragma link C++ class PacChromData-!;
//#pragma link C++ class PAC::LinearMapper-!;

#endif
