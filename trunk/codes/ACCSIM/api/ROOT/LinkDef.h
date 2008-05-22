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
#pragma link C++ namespace ACCSIM;
#pragma link C++ nestedclasses;

#pragma link C++ class ACCSIM::BasicPropagator-!;
#pragma link C++ class ACCSIM::RandomNumberGenerator-!;
#pragma link C++ class ACCSIM::GaussianGenerator-!;
#pragma link C++ class ACCSIM::OrbitUniformGenerator-!;
#pragma link C++ class ACCSIM::OrbitGaussianGenerator-!;
#pragma link C++ class ACCSIM::UniformGenerator-!;
#pragma link C++ class ACCSIM::TeapotGenerator-!;
//#pragma link C++ class ACCSIM::BunchGenerator-!;
//#pragma link C++ class ACCSIM::BunchAnalyzer-!;
#pragma link C++ class ACCSIM::CollimatorTracker-!;
#pragma link C++ class ACCSIM::IMaterialPropagator-!;
#pragma link C++ class ACCSIM::NuclearInteraction-!;
#pragma link C++ class ACCSIM::BetheBlochSlower-!;
#pragma link C++ class ACCSIM::FermiScatter-!;

#endif
