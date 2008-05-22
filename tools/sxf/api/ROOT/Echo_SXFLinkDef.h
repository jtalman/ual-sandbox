//A linkDef file is needed because of the namespace.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//  these lines deal with the namespace
#pragma link C++ namespace SXF;
#pragma link C++ nestedclasses;

// This line is also default for this class.
// the + tells rootcint to make the streamer.
#pragma link C++ class SXF::EchoAcceleratorReader-!;
#pragma link C++ class SXF::EchoElemBucket-!;
#pragma link C++ class SXF::EchoElemBucketRegistry-!;
#pragma link C++ class SXF::EchoElemError-!;
#pragma link C++ class SXF::EchoElement-!;
#pragma link C++ class SXF::EchoError-!;
#pragma link C++ class SXF::EchoNodeRegistry-!;
#pragma link C++ class SXF::EchoParser-!;
#pragma link C++ class SXF::EchoSequence-!;

#endif
