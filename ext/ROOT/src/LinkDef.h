//A linkDef file is needed because of the namespace.
//This file needs to be modifiled to include any new classes
//that are in the $(UAL_EXTRA)/ROOT/src directory.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//  these lines deal with the namespaces
#pragma link C++ namespace UAL;
#pragma link C++ namespace ZLIB;
#pragma link C++ namespace PAC;
//#pragma link C++ namespace TEAPOT;
//#pragma link C++ namespace SXF; 
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

/* These lines make public certain classes in $(UAL_EXTRA)/ROOT/src/*.hh
    the + tells rootcint to make the streamer.
    the - tells rootcint NOT to make the streamer. (mutually exclusive to +)
    the ! tells rootcint NOT to make the >> operator.
    See ROOT users Guide pg.296 for more information */
#pragma link C++ class UAL::RootShell+;
#pragma link C++ class UAL::LostTree-;
#pragma link C++ function UAL::bunch2Ntuple(const PAC::Bunch&, TNtupleD&);

#endif
