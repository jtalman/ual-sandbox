//A linkDef file is needed because of the namespace.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//  these lines deal with the namespace
#pragma link C++ namespace UAL;
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

// This line is also default for this class.
// the + tells rootcint to make the streamer.
#pragma link C++ class UAL::Object-!;
#pragma link C++ class UAL::AttributeSet-!;
#pragma link C++ class UAL::Algorithm-!;
#pragma link C++ class UAL::Element-!;
#pragma link C++ class UAL::Probe-!;
#pragma link C++ class UAL::RCObject-!;
#pragma link C++ class UAL::PropagatorNode-!;
#pragma link C++ class UAL::PropagatorNodePtr-!;
#pragma link C++ class UAL::PropagatorComponent-!;
#pragma link C++ class UAL::PropagatorSequence-!;
#pragma link C++ class UAL::AcceleratorPropagator-!;
#pragma link C++ class UAL::PropagatorFactory-!;
#pragma link C++ class UAL::AcceleratorNode-!;
#pragma link C++ class UAL::AcceleratorNodeFinder-!;
#pragma link C++ class UAL::APDF_Builder-!;


#endif
