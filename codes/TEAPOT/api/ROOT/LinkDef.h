//A linkDef file is needed because of the namespace.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// This line is also default for this class.
// the + tells rootcint to make the streamer.

//  these lines deal with the namespace
#pragma link C++ namespace TEAPOT;
#pragma link C++ namespace PAC;
#pragma link C++ namespace UAL;
#pragma link C++ nestedclasses;

#pragma link C++ class TEAPOT::LostCollector-!;



#endif
