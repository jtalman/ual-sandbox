#include "timer.h"


//UAL includes
#include "Templates/PacVector.h"
#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacLattice.h"
//#include "SMF/PacSmf.h" 

#include "ZLIB/Tps/Space.hh"
#include "ACCSIM/Collimator/CollimatorTracker.hh"
//#include "ual_sxf/Parser.hh"
#include "TEAPOT/Integrator/LostCollector.hh"

//ROOT includes
#include "TRandom3.h"
#include "TFile.h"

// UAL/ROOT includes
#include "Converters.h"
#include "LostTree.hh"
#include "RootShell.hh"


R__EXTERN TRandom *gRandom;
using namespace std;

int main(){

  //ZLIB::Space space(6, 5);
   UAL::RootShell shell;
   ACCSIM::CollimatorTracker* col;
   list<UAL::PropagatorNodePtr>::iterator ic;

  int i;

   // ************************************************************************
  cout << "\n1. Lattice Part." << endl;
  // ************************************************************************

  //  char sxfFile[]="./../data/ProtoBlue2004.sxf"; // "blue.sxf";
  // char sxfFile[]="./../data/blue-04-top114535-ideal.sxf";
  char sxfFile[]="./../data/blue.apertures.sxf";
  char outdir[] = "./out/";
  
  cout << " 1.2 Build the Accelerator Object from the SXF file: " 
	    << sxfFile << endl;
  
  shell.readSXF(sxfFile ,outdir);

 
  // ************************************************************************
  cout << "\n2. Beam Part Part I." << endl;
  // ************************************************************************

  int bunchsize=100;

  PAC::Bunch bunch(bunchsize); 
  //  bunch.setEnergy(250.0);
  // bunch.setMass(0.9382796);
  //bunch.setCharge(1.0);

  shell.setBeamAttributes(250.0, 0.9382796, 1.0);

  // ************************************************************************
  cout << "3. Algorithm Part. " << endl;
  // ************************************************************************

  char xmlFile[] = "../data/tracker.apdf";


  shell.readAPDF(xmlFile);
  UAL::AcceleratorPropagator* ap=shell.getPropagator();

  // ************************************************************************
  cout << "\n4. Collimator attributes. " << endl;
  // ************************************************************************

  list<UAL::PropagatorNodePtr> collTrackers = ap->getNodesByName("bi8-c");
  cout << "\nNumber of collimator trackers = " << collTrackers.size() << endl;

  if(collTrackers.size()!=0){
    float radlength=716.4*63.546/(29*30*log(287/sqrt(29.0)))/8.920*100;
    
    for(ic = collTrackers.begin(); ic != collTrackers.end(); ic++){
      col = dynamic_cast<ACCSIM::CollimatorTracker*>( (*ic).getPointer());
      col->setMaterial(63.546, 29, 8.920, radlength);
      col=0;
    }
    //set aperture of primary collimator
    col = dynamic_cast<ACCSIM::CollimatorTracker*>( (*collTrackers.begin()).getPointer());
    col->setAperture(ACCSIM::CollimatorTracker::XFLAT,-0.005,1);
    ic = collTrackers.end();
  }
  
  // ************************************************************************
  cout << "\n5. Test of APDF-based tracker. " << endl;
  // ************************************************************************

  char accName[] = "blue";

  shell.use(accName);
  PacLattice *lattice = (PacLattice *)shell.GetLattice();

  double t; // time variable
  int lsize = lattice->size();
  

  cout << "\nTeapot Tracker " << endl;
  cout << "size : " << lattice->size() << " elements " <<  endl;

  PacVector<int> indexes = lattice->indexes("col");
  int firstcol=indexes[0];
  
  cout<<"There are "<<indexes.size()<<" collimators"<<endl;
  cout << "The first collimator is at index "<<firstcol<<endl;

  // ************************************************************************
  cout << "\n6. Beam Part Part II - The Distribution." << endl;
  // ************************************************************************

  shell.generateBunch(bunch,firstcol);


  TNtupleD input("input","Input particle distribution","x:x':y:y':ct:dp:flag");
  UAL::bunch2Ntuple(bunch, input);

  // ************************************************************************
  cout << "\n7. Tracking Part" << endl;
  // ************************************************************************
  
  int nturns=1;
  TNtupleD output("output","Output of Tracked Particles","x:x':y:y':ct:dp:flag");
  TEAPOT::LostCollector& TheLost= TEAPOT::LostCollector::GetInstance();
  TheLost.RegisterBunch(bunch);
  
  shell.multitrack(bunch,nturns,firstcol);
  
  
  
  
  // ************************************************************************
  cout << "\n7. Analysis Part" << endl;
  // ************************************************************************
  
  TFile *file;
  file = new TFile("Test.root","NEW","",9);
  UAL::LostTree* Dead=new UAL::LostTree("Dead","Lost Particles during simulation");
  Dead->RegisterLattice(lattice);
  
  input.Write();
  Dead->Write();
  
  UAL::bunch2Ntuple(bunch, output);
  output.Write();

  //  cout<<"About to clean up dead"<<endl;
  delete Dead;
  //  cout<<"About to clean up file"<<endl;
  delete file;
  
  //  cout<<"About to leave"<<endl;
  
  return 1;
  
}
