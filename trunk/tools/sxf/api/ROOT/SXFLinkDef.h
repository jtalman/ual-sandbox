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
#pragma link C++ struct SXF_Key-!;
#pragma link C++ class SXF::OStream-!;
#pragma link C++ class SXF::ElemBucket-!;
#pragma link C++ class SXF::ElemBucketRegistry-!;
#pragma link C++ class SXF::AcceleratorNode-!;
#pragma link C++ class SXF::Element-!;
#pragma link C++ class SXF::Sequence-!;
#pragma link C++ class SXF::SequenceStack-!;
#pragma link C++ class SXF::NodeRegistry-!;
#pragma link C++ class SXF::AcceleratorReader-!;

#pragma link C++ class SXF::ElemBucketHash-!;
#pragma link C++ class SXF::ElemAlignHash-!;
#pragma link C++ class SXF::ElemApertureHash-!;
#pragma link C++ class SXF::ElemBeamBeamHash-!;
#pragma link C++ class SXF::ElemBendHash-!;
#pragma link C++ class SXF::ElemElSeparatorHash-!;
#pragma link C++ class SXF::ElemMultipoleHash-!;
#pragma link C++ class SXF::ElemRfCavityHash-!;
#pragma link C++ class SXF::ElemSolenoidHash-!;
#pragma link C++ class SXF::ElemEmptyHash-!;
#pragma link C++ class SXF::ElemCollimatorHash-!;

 
#endif
