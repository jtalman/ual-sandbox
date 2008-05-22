//A linkDef file is needed because of the namespace.
//This file needs to be modifiled to include any new classes
//that are in the $(UAL_EXTRA)/ROOT/src directory.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace CORAANT;
#pragma link C++ nestedclasses;

/* These lines make public certain classes in $(UAL_EXTRA)/ROOT/src/*.hh
    the + tells rootcint to make the streamer.
    the - tells rootcint NOT to make the streamer. (mutually exclusive to +)
    the ! tells rootcint NOT to make the >> operator.
    See ROOT users Guide pg.296 for more information */

#pragma link C++ class CORAANT::TGraphAdd!;
#pragma link C++ class CORAANT::TGTree+;
#pragma link C++ class CORAANT::TGChain-;
#pragma link C++ class CORAANT::TGNtuple-;
#pragma link C++ class CORAANT::TGNtupleD-;
#pragma link C++ class CORAANT::TTurnPlot+;
#pragma link C++ class CORAANT::TTuneGraph+;
#pragma link C++ class CORAANT::OneDFFT-!;
#pragma link C++ class CORAANT::real1DFFT-!;
#pragma link C++ class CORAANT::complex1DFFT-!;
#pragma link C++ class CORAANT::TTuneSpace+;


#endif
