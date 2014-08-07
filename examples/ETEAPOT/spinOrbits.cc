#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "Main/Eteapot.h"

#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "Optics/PacTMap.h"
#include "Integrator/TeapotElemBend.h"

#include "positionPrinter.hh"
#include "xmgracePrint.hh"

#include "ETEAPOT_MltTurn/Integrator/DipoleTracker.hh"
//#include "ETEAPOT_MltTurn/Integrator/DipoleAlgorithm.icc"
#include "ETEAPOT_MltTurn/Integrator/MltTracker.hh"
//#include "ETEAPOT_MltTurn/Integrator/DriftTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/MarkerTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/RFCavityTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 int startTime, endTime, totalTime;
// std::cerr << "ETEAPOT_MltTurn::DipoleTracker::m_m " << ETEAPOT_MltTurn::DipoleTracker::m_m << "\n";
// std::cerr << "ETEAPOT_MltTurn::MltTracker::m_m    " << ETEAPOT_MltTurn::MltTracker::m_m    << "\n";

 if(argc!=8){
  std::cout << "usage: ./determineTwiss ./data/E_BM_M1.0_sl4.sxf -1 40 0 5 0.01 10 (>&! OUT)\n";
  std::cout << "argv[0] is this executable         - ./determineTwiss               \n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_BM_M1.0_sl4.sxf       \n";
  std::cout << "argv[2] is the nominal electrode m - +1                             \n";
  std::cout << "argv[3] is the nominal electrode bend radius - 40 =.7854/.0196      \n";
  std::cout << "argv[4] is the initialSpin file creation type - 0                   \n";
  std::cout << "argv[5] is the number of turns - 5                                  \n";
  std::cout << "argv[6] is the fringe field length - 0.01                           \n";
  std::cout << "argv[7] is the ""decimation factor"" - 10                           \n";
  exit(0);
 }

 ofstream m_m;
 m_m.open ("m_m");
 m_m << argv[2];
 m_m.close();
 ETEAPOT_MltTurn::DipoleTracker* edt=new ETEAPOT_MltTurn::DipoleTracker();
 ETEAPOT_MltTurn::MltTracker*    mdt=new ETEAPOT_MltTurn::MltTracker();
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::dZFF " << ETEAPOT_MltTurn::DipoleTracker::dZFF << "\n";
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::m_m " << ETEAPOT_MltTurn::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT_MltTurn::MltTracker::m_m    " << ETEAPOT_MltTurn::MltTracker::m_m    << "\n";
               ETEAPOT_MltTurn::DipoleTracker::dZFF=atof( argv[6] );
               ETEAPOT_MltTurn::DipoleTracker::m_m=atof( argv[2] );
               ETEAPOT_MltTurn::MltTracker::m_m   =atof( argv[2] );
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::dZFF " << ETEAPOT_MltTurn::DipoleTracker::dZFF << "\n";
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::m_m " << ETEAPOT_MltTurn::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT_MltTurn::MltTracker::m_m    " << ETEAPOT_MltTurn::MltTracker::m_m    << "\n";

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "designBeamValues.hh"
 #include "setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();
 std::cerr << "ba.getG() "  << ba.getG()  << "\n";
 std::cerr << "ba.get_g() " << ba.get_g() << "\n";
 #include "extractParameters.h"
// #include "probeDataForTwiss"
#define TINY 1E-6
double x1typ  = TINY;
double x2typ  = TINY;
double y1typ  = TINY, x3typ  = TINY;
double y2typ  = TINY, x4typ  = TINY;
double x5typ  = TINY;
double deltyp = TINY, x6typ  = TINY;
  #include "userBunch"
//#include "trtrin"
 #include "simulatedProbeValues"

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

 shell.setMapAttributes(UAL::Args() << UAL::Arg("order", order));

 // ************************************************************************
 std::cout << "\nBuild lattice." << std::endl;
 // ************************************************************************

 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

 // ************************************************************************
 std::cout << "\nAdd split ." << std::endl;
 // ************************************************************************
  
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", splitForBends));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", splitForQuads));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", splitForSexts));

 // ************************************************************************
 std::cout << "Select lattice." << std::endl;
 // ************************************************************************

 shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

 // ************************************************************************
 std::cout << "\nWrite SXF file ." << std::endl;
 // ************************************************************************

 shell.writeSXF(UAL::Args() << UAL::Arg("file",  outputFile.c_str()));

 // ************************************************************************
 std::cout << "\nDefine beam parameters." << std::endl;
 // ************************************************************************

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
 // ************************************************************************
 std::cout << "\nAlgorithm Part. " << std::endl;
 // ************************************************************************

 UAL::APDF_Builder apBuilder;

 apBuilder.setBeamAttributes(ba);

 UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

 if(ap == 0) {
   std::cout << "Accelerator Propagator has not been created " << std::endl;
   return 1;
 }

// ETEAPOT_MltTurn::MltAlgorithm<double, PAC::Position>::m_sxfFilename = argv[1];
 OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
 Teapot* teapot = optics.m_teapot;
 PacSurveyData surveyData;
 char drift[13]="Drift       ";
 std::string nameInput,nameOutput;
 std::string typeInput,typeOutput;
 std::string designNameInput;
 std::cerr << "teapot->size() " << teapot->size() << "\n";
 double xX,yX,zX,sX;
 int mltK=0,drft=0,bend=0,mark=0,RF=0;

 int nonDrifts=0;

 double sPrevious=0;
 double sBndDlta;
 double totSplits;

 for(int i = 0; i < teapot->size(); i++){
  TeapotElement& te = teapot->element(i);
  nameInput=te.getDesignName();
  typeInput=te.getType();
  designNameInput=te.getDesignName();
  teapot->survey(surveyData,i,i+1);

  if(nameInput.length()==1 ){nameInput+="           "; }
  if(nameInput.length()==2 ){nameInput+="          ";  }
  if(nameInput.length()==3 ){nameInput+="         ";   }   
  if(nameInput.length()==4 ){nameInput+="        ";    }   
  if(nameInput.length()==5 ){nameInput+="       ";     }   
  if(nameInput.length()==6 ){nameInput+="      ";      }   
  if(nameInput.length()==7 ){nameInput+="     ";       }   
  if(nameInput.length()==8 ){nameInput+="    ";        }   
  if(nameInput.length()==9 ){nameInput+="   ";         }   
  if(nameInput.length()==10){nameInput+="  ";          }   
  if(nameInput.length()==11){nameInput+=" ";           }   
  if(nameInput.length()==12){nameInput+="";            }   

  if(nameInput.size()>=1){
   nameOutput=nameInput;
  }
  else{
   nameOutput=drift;
  }

  if(typeInput.length()==1 ){typeInput+="           "; }
  if(typeInput.length()==2 ){typeInput+="          ";  }
  if(typeInput.length()==3 ){typeInput+="         ";   }   
  if(typeInput.length()==4 ){typeInput+="        ";    }   
  if(typeInput.length()==5 ){typeInput+="       ";     }   
  if(typeInput.length()==6 ){typeInput+="      ";      }   
  if(typeInput.length()==7 ){typeInput+="     ";       }   
  if(typeInput.length()==8 ){typeInput+="    ";        }   
  if(typeInput.length()==9 ){typeInput+="   ";         }   
  if(typeInput.length()==10){typeInput+="  ";          }   
  if(typeInput.length()==11){typeInput+=" ";           }   
  if(typeInput.length()==12){typeInput+="";            }   

  if(typeInput.size()>=1){
   typeOutput=typeInput;
  }
  else{
   typeOutput=drift;
  }

  xX = surveyData.survey().x();
  yX = surveyData.survey().y();
  zX = surveyData.survey().z();
  sX = surveyData.survey().suml();

  if( typeOutput=="Quadrupole  "){
   nonDrifts++;
   std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
   ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_elementName[mltK]=nameOutput;
   ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_sX[mltK++]=sX;
   sPrevious=sX;
  }

  if( typeOutput=="Sextupole   "){
   nonDrifts++;
   std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
   ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_elementName[mltK]=nameOutput;
   ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_sX[mltK++]=sX;
   sPrevious=sX;
  }

  if( typeOutput=="Drift       "){
   std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
// ETEAPOT_MltTurn::DriftAlgorithm<double,PAC::Position>::drft_m_elementName[drft]=nameOutput;
// ETEAPOT_MltTurn::DriftAlgorithm<double,PAC::Position>::drft_m_sX[drft++]=sX;
   sPrevious=sX;
  }

  if( typeOutput=="Sbend       "){
// nonDrifts++;
   totSplits=2*pow(2,splitForBends);
// std::cerr << "totSplits " << totSplits << "\n";
   sBndDlta=(sX-sPrevious)/totSplits;
// sBndDlta=(sX-sPrevious)/(1+splitForBends);
   for(int j=0;j<totSplits;j++){
    std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sPrevious+sBndDlta << "\n";
//  std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
    DipoleAlgorithm<double,PAC::Position>::bend_m_elementName[bend]=nameOutput;
    DipoleAlgorithm<double,PAC::Position>::bend_m_sX[bend++]=sPrevious+sBndDlta;
    nonDrifts++;
    sPrevious+=sBndDlta;
//  algorithm<double,PAC::Position>::bend_m_sX[bend++]=sX;
   }
  }

  if( typeOutput=="Marker      "){
   nonDrifts++;
   std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
   ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[mark]=nameOutput;
   ETEAPOT_MltTurn::MarkerTracker::Mark_m_sX[mark++]=sX;
   sPrevious=sX;
  }

  if( typeOutput=="RfCavity    "){
   nonDrifts++;
   std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
   ETEAPOT_MltTurn::RFCavityTracker::RF_m_elementName[RF]=nameOutput;
   ETEAPOT_MltTurn::RFCavityTracker::RF_m_sX[RF++]=sX;
   sPrevious=sX;
  }

//std::cerr << "name " << nameOutput << " type " << typeOutput << " " << xX << " " << yX << " " << zX << " " << sX << "\n";
 }

std::cerr << "teapot->size() " << teapot->size() << "\n";
std::cerr << "nonDrifts      " << nonDrifts      << "\n";

/*
 for(int i = 0; i < teapot->size(); i++){
  std::cerr << "applyMltKick ETEAPOT_MltTurn::DriftAlgorithm::drft_m_elementName[" << i << "]: "   << ETEAPOT_MltTurn::DriftAlgorithm<double,PAC::Position>::drft_m_elementName[i]   << " " << ETEAPOT_MltTurn::DriftAlgorithm<double,PAC::Position>::drft_m_sX[i] << "\n";
 }
 for(int i = 0; i < teapot->size(); i++){
  std::cerr << "applyMltKick ETEAPOT_MltTurn::MltAlgorithm::m_elementName[       " << i << "]: "   << ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_elementName[i]      << " " << ETEAPOT_MltTurn::MltAlgorithm<double,PAC::Position>::Mlt_m_sX[i] << "\n";
 }
 for(int i = 0; i < teapot->size(); i++){
  std::cerr << "applyMltKick algorithm:bend_:m_elementName[              " << i << "]: "   << algorithm<double,PAC::Position>::bend_m_elementName[i]                 << " " << algorithm<double,PAC::Position>::bend_m_sX[i] << "\n";
 }
*/

 double a0x=     (double)0;
 double b0x=     (double)0;
 double mu_xTent=(double)0;
 double a0y=     (double)0;
 double b0y=     (double)0;
 double mu_yTent=(double)0;
//int turns=1;
// turns=1;
  turns=atoi( argv[5] );
//#include "probeDataForTwiss"
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::dZFF " << ETEAPOT_MltTurn::DipoleTracker::dZFF << "\n";
 std::cerr << "ETEAPOT_MltTurn::DipoleTracker::m_m " << ETEAPOT_MltTurn::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT_MltTurn::MltTracker::m_m    " << ETEAPOT_MltTurn::MltTracker::m_m    << "\n";

#include"S"
//char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};
  #include "spin"

//ETEAPOT_MltTurn::MarkerTracker::initialize();
//#include"../../codes/ETEAPOT_MltTurn/src/ETEAPOT_MltTurn/Integrator/setMarkerTrackerSpin"
//#include"setMarkerTrackerSpin"
//#include"setDipoleAlgorithmSpin"
ETEAPOT_MltTurn::DipoleTracker::initialize();
ETEAPOT_MltTurn::MltTracker::initialize();

//#include"verifyMarkerTrackerSpin"
#include"verifyDipoleTrackerSpin"
#include"verifyMltTrackerSpin"

// ofstream iS;
// iS.open ("initialSpin");    // for server side compatibility
                             // Full spin functionality is anticipated in 
                             // orbitsWithSpin.cc or such
// iS.close();

 positionPrinter pP; 
 pP.open( "NikolayOut" );
// pP.open(orbitFile.c_str());

// xmgracePrint xP;
// xP.open("bunchSub0");

 ba.setElapsedTime(0.0);

int decFac=atoi(argv[7]);
startTime = time(NULL);
 for(int iturn = 0; iturn <= (turns-1); iturn++){
//ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   if( iturn%decFac == 0 ){
    pP.write(iturn, ip, bunch);
   }
// xP.write(iturn, ip, bunch);
  }
  ap -> propagate(bunch);
 }
endTime = time(NULL);
totalTime = endTime - startTime;
std::cerr << "Runtime: " << totalTime << " seconds\n";

 pP.close();

/*
 ofstream markerSpin;
 markerSpin.open ("out/VERIF/markerSpin");
 markerSpin << setiosflags( ios::showpos    );
 markerSpin << setiosflags( ios::uppercase  );
 markerSpin << setiosflags( ios::scientific );
 markerSpin << setfill( ' ' );
 markerSpin << setiosflags( ios::left );
 markerSpin << setprecision(13) ;

 for(int iq=0;iq<=19;iq++){
  markerSpin << S[iq] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][0] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][1] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][2] << "\n";
 }
 markerSpin << S[iq] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][0] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][1] << " " << ETEAPOT_MltTurn::MarkerTracker::spin[iq][2];
 markerSpin.close();
*/

 return (int)0;
}
