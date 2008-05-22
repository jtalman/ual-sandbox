#include <iostream>
#include <pcre.h>

#include "UAL/Common/Def.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "Optics/PacTMap.h"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "ZLIB/Tps/Space.hh"
#include "ual_sxf/Parser.hh"

int main(){

  ZLIB::Space space(6, 5);

   // ************************************************************************
  std::cout << "\n1. Read the SXF file" << std::endl;
  // ************************************************************************

  string sxfFile = "rhic_injection.sxf";

  string inFile   = "./../data/"; inFile   += sxfFile; 
  string echoFile = "./out/";     echoFile += sxfFile; echoFile += ".echo";
  string outFile =  "./out/";     outFile  += sxfFile; outFile  += ".out";

  UAL_SXF_Parser parser;
  parser.read(inFile.data(), echoFile.data()); 
  parser.write(outFile.data());

  // ************************************************************************
  std::cout << "\n2. Select a lattice for operations" << std::endl;
  // ************************************************************************

  string accName = "blue";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    exit(1);
  }

  PacLattice lattice = *latIterator;

  // ************************************************************************
  std::cout << "\n3. Define beam parameters." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes ba;
  ba.setEnergy(250.0);

  // ************************************************************************
  std::cout << "\n4. Find closed orbit and Twiss parameters." << std::endl;
  // ************************************************************************

  Teapot teapot(lattice);

  // Find closed orbit

  PAC::Position clorbit;
  teapot.clorbit(clorbit, ba);

  std::cout << "closed orbit "
	    << " : x = " << clorbit.getX() 
	    << ", px = " << clorbit.getPX() 
	    << ", y = "  << clorbit.getY() 
	    << ", py = " << clorbit.getPY() 
	    << endl;

  // Find Twiss parameters in the injection point

  PacTwissData tw;
  teapot.twiss(tw, ba, clorbit);

  // ************************************************************************
  std::cout << "\n5. Make regular expression for selecting elements." << std::endl;
  // ************************************************************************

  std::string strPattern  = "Quadrupole"; // "^(q.*)";  

  const char *error;
  int erroffset;
  int overtor[30];   
  pcre* rePattern         = pcre_compile(strPattern.data(), 0, &error, &erroffset, NULL);
  pcre_extra* pePattern   = pcre_study(rePattern, 0, &error);

  // ************************************************************************
  std::cout << "\n6. Propagate Twiss parameters and print them for selected elements." << std::endl;
  // ************************************************************************

  ofstream outfile;
  outfile.open("quads.out");

  double at  = 0;
  tw.mu(0, 0.0);
  double mux = 0.0;

  tw.mu(1, 0.0);
  double muy = 0.0;

  int counter = 0;
  for(int i = 0; i < lattice.getNodeCount(); i++){

     UAL::AcceleratorNode* const anode = lattice.getNodeAt(i);  

     // Check the element design name

     std::string elname = anode->getDesignName();
     // int rc = pcre_exec(rePattern, pePattern, elname.data(), elname.size(), 
     // 0, 0, overtor, 30);
     std::string eltype = anode->getType();
     
     // if(rc > 1){
     // if(eltype == "Quadrupole") {
     if(eltype == "Vmonitor" || eltype == "Hmonitor" ) {
       outfile   << counter++ << " " 
		 << i << " " 
		 << elname << " " 
		 << at << " " 
		 << tw.mu(0)/UAL::pi/2.0 << " " 
		 << tw.beta(0) << "\n";
     }

     // find element map

     PacTMap sectorMap(6);
     sectorMap.refOrbit(clorbit);    
     teapot.trackMap(sectorMap, ba, i, i+1);

     // propagate Twiss parameters through this map

     teapot.trackTwiss(tw, sectorMap);

     // check mu

     if((tw.mu(0) - mux) < 0.0 ) { tw.mu(0, tw.mu(0) + 1.0); }
     mux = tw.mu(0);

     if((tw.mu(1) - muy) < 0.0 ) { tw.mu(1, tw.mu(1) + 1.0); }
     muy = tw.mu(1);

     // propagate closed orbit

     teapot.trackClorbit(clorbit, ba, i, i+1);    

     // update position

     at += anode->getLength();
  }

  outfile.close();
}



