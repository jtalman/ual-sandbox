#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 ofstream ofstrm("flat_sxf");
 if (!ofstrm){
  cout << "Couldn’t open file: " << "flat_sxf" << endl;
 }
 else{
  cout << "Found and opened file: " << "flat_sxf" << endl;
 }
 ofstrm << setiosflags( ios::showpos   );
 ofstrm << setiosflags( ios::uppercase );
 ofstrm << setiosflags( ios::scientific );
 ofstrm << setw( 11 );
 ofstrm << setfill( ' ' );
 ofstrm << setiosflags( ios::right );
 ofstrm << setprecision(4) ;

 UAL::Shell shell;
 std::string sxfFile = argv[1];
 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));
 shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

/*
 std::cout << " calculate suml" << std::endl;
 shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));
 shell.analysis(UAL::Args());
*/

 OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
 Teapot* teapot = optics.m_teapot;
 std::cout << " teapot->size() " << teapot->size() << std::endl;

 double R0=40;

 PacSurveyData surveyData;

 char drift[12]="drift      ";
 std::string nameInput;
 std::string nameOutput;
 double xN = 0;
 double yN = 0;
 double zN = 0;
 double sN = 0;

 double xX;
 double yX;
 double zX;
 double sX;

 double a,b,c,d;
 double f;                                     // "alpha" -->> fuh
 double u;                                     // "beta"  -->> uh
 double g;                                     // gamma

 double AN,BN,CN;                              // "quadratic coefficients" via (a,0,b) element entry point
                                               // AN x^2 + BN x + CN = 0
 double AX,BX,CX;                              // AX x^2 + BX x + CX = 0   via (c,0,d) element  exit point

 double xCPN,yCPN=0,zCPN;                         // center via plus  quadratic solution and element entry point
 double xCPX,yCPX=0,zCPX;                         // center via plus  quadratic solution and element exit  point
 double xCMN,yCMN=0,zCMN;                         // center via minus quadratic solution and element entry point
 double xCMX,yCMX=0,zCMX;                         // center via minus quadratic solution and element exit  point

 int myIndex=-1;

 for(int i = 0; i < teapot->size(); i++){
  TeapotElement& te = teapot->element(i);
  nameInput=te.getDesignName();
/*
  if(nameInput=="mkb" ){nameInput+="  "; }
  if(nameInput=="mk1" ){nameInput+="  "; }
  if(nameInput=="mke" ){nameInput+="  "; }
  if(nameInput=="bpm" ){nameInput+="  "; }
  if(nameInput=="q1h" ){nameInput+="  "; }
  if(nameInput=="q1"  ){nameInput+="   ";}
  if(nameInput=="q2"  ){nameInput+="   ";}
  if(nameInput=="q3h" ){nameInput+="  "; }
  if(nameInput=="s1"  ){nameInput+="   ";}
  if(nameInput=="s2"  ){nameInput+="   ";}
  if(nameInput=="s3"  ){nameInput+="   ";}
  if(nameInput=="bend"){nameInput+=" ";  }
*/

/*
  if(nameInput=="mbegin"){nameInput+="     "; }    // ./data/E_FirstTest.sxf
  if(nameInput=="marcin"){nameInput+="     "; }
  if(nameInput=="mcellin"){nameInput+="    "; }
  if(nameInput=="sb"){nameInput+="         "; }
  if(nameInput=="qb"){nameInput+="         "; }    // JDT - diff
  if(nameInput=="bh"){nameInput+="         "; }
  if(nameInput=="sc"){nameInput+="         "; }
  if(nameInput=="qc"){nameInput+="         "; }
  if(nameInput=="qa"){nameInput+="         "; }
  if(nameInput=="sa"){nameInput+="         "; }
  if(nameInput=="mcellcenter"){nameInput+=""; }
  if(nameInput=="mcellout"){nameInput+="   "; }
  if(nameInput=="marcout"){nameInput+="    "; }
  if(nameInput=="mend"){nameInput+="       "; }
*/

  if(nameInput=="mbegin"){nameInput+="     "; }    // ./data/E_Kepler.sxf
  if(nameInput=="marcin"){nameInput+="     "; }
  if(nameInput=="mcellin"){nameInput+="    "; }
  if(nameInput=="q"){nameInput+="          "; }    // JDT - diff
  if(nameInput=="bh"){nameInput+="         "; }
  if(nameInput=="mcellcenter"){nameInput+=""; }
  if(nameInput=="mcellout"){nameInput+="   "; }
  if(nameInput=="marcout"){nameInput+="    "; }
  if(nameInput=="mend"){nameInput+="       "; }


/*
  if(nameInput=="mbegin"){nameInput+="    "; }
  if(nameInput=="marcin"){nameInput+="    "; }
  if(nameInput=="qch"){nameInput+="       "; }
  if(nameInput=="sc"){nameInput+="        "; }
  if(nameInput=="bh"){nameInput+="        "; }
  if(nameInput=="qbh"){nameInput+="       "; }
  if(nameInput=="sb"){nameInput+="        "; }
  if(nameInput=="qah"){nameInput+="       "; }
  if(nameInput=="sa"){nameInput+="        "; }
  if(nameInput=="mslndcent"){nameInput+=" "; }
  if(nameInput=="mhalf"){nameInput+="     "; }
  if(nameInput=="marcout"){nameInput+="   "; }
  if(nameInput=="mend"){nameInput+="      "; }
*/

/*                                                    COSY
  if(nameInput=="tum"){nameInput+="       "; }
  if(nameInput=="qf"){nameInput+="        "; }
  if(nameInput=="infr24"){nameInput+="    "; }
  if(nameInput=="qbe24i"){nameInput+="    "; }
  if(nameInput=="be24"){nameInput+="      "; }
  if(nameInput=="qbe24e"){nameInput+="    "; }
  if(nameInput=="oufr24"){nameInput+="    "; }
  if(nameInput=="qd"){nameInput+="        "; }
  if(nameInput=="qt1"){nameInput+="       "; }
  if(nameInput=="qt2"){nameInput+="       "; }
  if(nameInput=="qt3"){nameInput+="       "; }
  if(nameInput=="mx01"){nameInput+="      "; }
  if(nameInput=="qt4"){nameInput+="       "; }
  if(nameInput=="qt5"){nameInput+="       "; }
  if(nameInput=="qt6"){nameInput+="       "; }
  if(nameInput=="qt7"){nameInput+="       "; }
  if(nameInput=="mx02"){nameInput+="      "; }
  if(nameInput=="qt8"){nameInput+="       "; }
  if(nameInput=="qt9"){nameInput+="       "; }
  if(nameInput=="qt10"){nameInput+="      "; }
  if(nameInput=="mx03"){nameInput+="      "; }
  if(nameInput=="qt11"){nameInput+="      "; }
  if(nameInput=="qt12"){nameInput+="      "; }
  if(nameInput=="qt13"){nameInput+="      "; }
  if(nameInput=="mx04"){nameInput+="      "; }
  if(nameInput=="qt14"){nameInput+="      "; }
  if(nameInput=="qt15"){nameInput+="      "; }
  if(nameInput=="qt16"){nameInput+="      "; }
  if(nameInput=="infr1"){nameInput+="     "; }
  if(nameInput=="qbe1i"){nameInput+="     "; }
  if(nameInput=="be1"){nameInput+="       "; }
  if(nameInput=="qbe1e"){nameInput+="     "; }
  if(nameInput=="oufr1"){nameInput+="     "; }
  if(nameInput=="infr2"){nameInput+="     "; }
  if(nameInput=="qbe2i"){nameInput+="     "; }
  if(nameInput=="be2"){nameInput+="       "; }
  if(nameInput=="qbe2e"){nameInput+="     "; }
  if(nameInput=="oufr2"){nameInput+="     "; }
  if(nameInput=="rfsol"){nameInput+="     "; }
  if(nameInput=="infr3"){nameInput+="     "; }
  if(nameInput=="qbe3i"){nameInput+="     "; }
  if(nameInput=="be3"){nameInput+="       "; }
  if(nameInput=="qbe3e"){nameInput+="     "; }
  if(nameInput=="oufr3"){nameInput+="     "; }
  if(nameInput=="mx05"){nameInput+="      "; }
  if(nameInput=="infr4"){nameInput+="     "; }
  if(nameInput=="qbe4i"){nameInput+="     "; }
  if(nameInput=="be4"){nameInput+="       "; }
  if(nameInput=="qbe4e"){nameInput+="     "; }
  if(nameInput=="oufr4"){nameInput+="     "; }

  if(nameInput=="mx06"){nameInput+="      "; }

  if(nameInput=="qu5"){nameInput+="       "; }
  if(nameInput=="infr5"){nameInput+="     "; }
  if(nameInput=="qbe5i"){nameInput+="     "; }
  if(nameInput=="be5"){nameInput+="       "; }
  if(nameInput=="qbe5e"){nameInput+="     "; }
  if(nameInput=="oufr5"){nameInput+="     "; }

  if(nameInput=="qu6"){nameInput+="       "; }
  if(nameInput=="infr6"){nameInput+="     "; }
  if(nameInput=="qbe6i"){nameInput+="     "; }
  if(nameInput=="be6"){nameInput+="       "; }
  if(nameInput=="qbe6e"){nameInput+="     "; }
  if(nameInput=="oufr6"){nameInput+="     "; }

  if(nameInput=="mx07"){nameInput+="      "; }

  if(nameInput=="qu7"){nameInput+="       "; }
  if(nameInput=="infr7"){nameInput+="     "; }
  if(nameInput=="qbe7i"){nameInput+="     "; }
  if(nameInput=="be7"){nameInput+="       "; }
  if(nameInput=="qbe7e"){nameInput+="     "; }
  if(nameInput=="oufr7"){nameInput+="     "; }

  if(nameInput=="qu8"){nameInput+="       "; }
  if(nameInput=="infr8"){nameInput+="     "; }
  if(nameInput=="qbe8i"){nameInput+="     "; }
  if(nameInput=="be8"){nameInput+="       "; }
  if(nameInput=="qbe8e"){nameInput+="     "; }
  if(nameInput=="oufr8"){nameInput+="     "; }

  if(nameInput=="mx08"){nameInput+="      "; }

  if(nameInput=="qu9"){nameInput+="       "; }
  if(nameInput=="infr9"){nameInput+="     "; }
  if(nameInput=="qbe9i"){nameInput+="     "; }
  if(nameInput=="be9"){nameInput+="       "; }
  if(nameInput=="qbe9e"){nameInput+="     "; }
  if(nameInput=="oufr9"){nameInput+="     "; }

  if(nameInput=="mx09"){nameInput+="      "; }

  if(nameInput=="qu10"){nameInput+="      "; }
  if(nameInput=="infr10"){nameInput+="    "; }
  if(nameInput=="qbe10i"){nameInput+="    "; }
  if(nameInput=="be10"){nameInput+="      "; }
  if(nameInput=="qbe10e"){nameInput+="    "; }
  if(nameInput=="oufr10"){nameInput+="    "; }

  if(nameInput=="qu11"){nameInput+="      "; }
  if(nameInput=="infr11"){nameInput+="    "; }
  if(nameInput=="qbe11i"){nameInput+="    "; }
  if(nameInput=="be11"){nameInput+="      "; }
  if(nameInput=="qbe11e"){nameInput+="    "; }
  if(nameInput=="oufr11"){nameInput+="    "; }

  if(nameInput=="qu12"){nameInput+="      "; }
  if(nameInput=="infr12"){nameInput+="    "; }
  if(nameInput=="qbe12i"){nameInput+="    "; }
  if(nameInput=="be12"){nameInput+="      "; }
  if(nameInput=="qbe12e"){nameInput+="    "; }
  if(nameInput=="oufr12"){nameInput+="    "; }

  if(nameInput=="qt17"){nameInput+="      "; }
  if(nameInput=="qt18"){nameInput+="      "; }
  if(nameInput=="qt19"){nameInput+="      "; }
  if(nameInput=="mx10"){nameInput+="      "; }
  if(nameInput=="qt20"){nameInput+="      "; }

  if(nameInput=="rfcav"){nameInput+="     "; }

  if(nameInput=="qt21"){nameInput+="      "; }
  if(nameInput=="qt22"){nameInput+="      "; }
  if(nameInput=="qt23"){nameInput+="      "; }

  if(nameInput=="mx11"){nameInput+="      "; }

  if(nameInput=="qt24"){nameInput+="      "; }
  if(nameInput=="qt25"){nameInput+="      "; }

  if(nameInput=="mx12"){nameInput+="      "; }

  if(nameInput=="qt26"){nameInput+="      "; }
  if(nameInput=="qt27"){nameInput+="      "; }
  if(nameInput=="qt28"){nameInput+="      "; }
  if(nameInput=="qt29"){nameInput+="      "; }

  if(nameInput=="mx13"){nameInput+="      "; }

  if(nameInput=="qt30"){nameInput+="      "; }
  if(nameInput=="qt31"){nameInput+="      "; }
  if(nameInput=="qt32"){nameInput+="      "; }

  if(nameInput=="qu13"){nameInput+="      "; }
  if(nameInput=="infr13"){nameInput+="    "; }
  if(nameInput=="qbe13i"){nameInput+="    "; }
  if(nameInput=="be13"){nameInput+="      "; }
  if(nameInput=="qbe13e"){nameInput+="    "; }
  if(nameInput=="oufr13"){nameInput+="    "; }

  if(nameInput=="qu14"){nameInput+="      "; }
  if(nameInput=="infr14"){nameInput+="    "; }
  if(nameInput=="qbe14i"){nameInput+="    "; }
  if(nameInput=="be14"){nameInput+="      "; }
  if(nameInput=="qbe14e"){nameInput+="    "; }
  if(nameInput=="oufr14"){nameInput+="    "; }

  if(nameInput=="qu15"){nameInput+="      "; }
  if(nameInput=="infr15"){nameInput+="    "; }
  if(nameInput=="qbe15i"){nameInput+="    "; }
  if(nameInput=="be15"){nameInput+="      "; }
  if(nameInput=="qbe15e"){nameInput+="    "; }
  if(nameInput=="oufr15"){nameInput+="    "; }

  if(nameInput=="mx14"){nameInput+="      "; }

  if(nameInput=="qu16"){nameInput+="      "; }
  if(nameInput=="infr16"){nameInput+="    "; }
  if(nameInput=="qbe16i"){nameInput+="    "; }
  if(nameInput=="be16"){nameInput+="      "; }
  if(nameInput=="qbe16e"){nameInput+="    "; }
  if(nameInput=="oufr16"){nameInput+="    "; }

  if(nameInput=="mx15"){nameInput+="      "; }

  if(nameInput=="qu17"){nameInput+="      "; }
  if(nameInput=="infr17"){nameInput+="    "; }
  if(nameInput=="qbe17i"){nameInput+="    "; }
  if(nameInput=="be17"){nameInput+="      "; }
  if(nameInput=="qbe17e"){nameInput+="    "; }
  if(nameInput=="oufr17"){nameInput+="    "; }

  if(nameInput=="qu18"){nameInput+="      "; }
  if(nameInput=="infr18"){nameInput+="    "; }
  if(nameInput=="qbe18i"){nameInput+="    "; }
  if(nameInput=="be18"){nameInput+="      "; }
  if(nameInput=="qbe18e"){nameInput+="    "; }
  if(nameInput=="oufr18"){nameInput+="    "; }

  if(nameInput=="mx16"){nameInput+="      "; }

  if(nameInput=="qu19"){nameInput+="      "; }
  if(nameInput=="infr19"){nameInput+="    "; }
  if(nameInput=="qbe19i"){nameInput+="    "; }
  if(nameInput=="be19"){nameInput+="      "; }
  if(nameInput=="qbe19e"){nameInput+="    "; }
  if(nameInput=="oufr19"){nameInput+="    "; }

  if(nameInput=="qu20"){nameInput+="      "; }
  if(nameInput=="infr20"){nameInput+="    "; }
  if(nameInput=="qbe20i"){nameInput+="    "; }
  if(nameInput=="be20"){nameInput+="      "; }
  if(nameInput=="qbe20e"){nameInput+="    "; }
  if(nameInput=="oufr20"){nameInput+="    "; }

  if(nameInput=="mx17"){nameInput+="      "; }

  if(nameInput=="qu21"){nameInput+="      "; }
  if(nameInput=="infr21"){nameInput+="    "; }
  if(nameInput=="qbe21i"){nameInput+="    "; }
  if(nameInput=="be21"){nameInput+="      "; }
  if(nameInput=="qbe21e"){nameInput+="    "; }
  if(nameInput=="oufr21"){nameInput+="    "; }

  if(nameInput=="mx18"){nameInput+="      "; }

  if(nameInput=="qu22"){nameInput+="      "; }
  if(nameInput=="infr22"){nameInput+="    "; }
  if(nameInput=="qbe22i"){nameInput+="    "; }
  if(nameInput=="be22"){nameInput+="      "; }
  if(nameInput=="qbe22e"){nameInput+="    "; }
  if(nameInput=="oufr22"){nameInput+="    "; }

  if(nameInput=="qu23"){nameInput+="      "; }
  if(nameInput=="infr23"){nameInput+="    "; }
  if(nameInput=="qbe23i"){nameInput+="    "; }
  if(nameInput=="be23"){nameInput+="      "; }
  if(nameInput=="qbe23e"){nameInput+="    "; }
  if(nameInput=="oufr23"){nameInput+="    "; }

  if(nameInput=="endm"){nameInput+="      "; }
*/

  myIndex++;

  if(nameInput.size()>=1){
   nameOutput=nameInput;
  }
  else{
   nameOutput=drift;
  }

  teapot->survey(surveyData,i,i+1);

  xX = surveyData.survey().x();
  yX = surveyData.survey().y();
  zX = surveyData.survey().z();
  sX = surveyData.survey().suml();

  a=xN,b=zN;
  c=xX   ,d=zX   ;
  f=c-a,  u=d-b;
  g=-(a*a+b*b-c*c-d*d)/2;

  AN=1+f*f/u/u;
  BN=-2*(a+g*f/u/u-f/u);                                   // ANx^2 + BNx + CN = 0
  CN=a*a+g*g/u/u-2*g/u+b*b-R0*R0;                          // via entry point

  AX=1+f*f/u/u;
  BX=-2*(c+g*f/u/u-f/u);                                   // AXx^2 + BXx + CX = 0
  CX=c*c+g*g/u/u-2*g/u+d*d-R0*R0;                          // via  exit point

  xCPN=(-2*BN+sqrt(BN*BN-4*AN*CN))/2/AN;  zCPN=(g-xCPN*f)/u;  // plus  quadratic solution via entry point
  xCMN=(-2*BN-sqrt(BN*BN-4*AN*CN))/2/AN;  zCMN=(g-xCMN*f)/u;  // minus quadratic solution via entry point

  xCPX=(-2*BX+sqrt(BX*BX-4*AX*CX))/2/AX;  zCPX=(g-xCPX*f)/u;  // plus  quadratic solution via  exit point
  xCMX=(-2*BX-sqrt(BX*BX-4*AX*CX))/2/AX;  zCMX=(g-xCMX*f)/u;  // minus quadratic solution via  exit point

  ofstrm
         << setw( 5 )
         << i << " "
         << setw( 11 )
         << nameOutput << " "
         << setw( 11 )
         << sN << " ( "
    
         << setw( 11 )
         << xN << " , "
//       << xX << " "
         << setw( 11 )
         << yN << " , "
//       << yX << " "
         << setw( 11 )
         << zN << " )"
//       << zX << " "
//       << nameOutput << " "
//       << setw( 11 )
//       << sN
//       << sX
         << "-( "
         << setw( 11 )
         << xX << " , "
         << setw( 11 )
         << yX << " , "
         << setw( 11 )
         << zX << " ) "
         << sX << " "
         << sX-sN
         << std::endl;
    
  xN=xX;
  yN=yX;
  zN=zX;
  sN=sX;
 }

 std::cout << "\nOne lattice element surveyed at a time suml():" << "\n";
 std::cout << "       surveyData.survey().suml() " << surveyData.survey().suml()     << "\n";

 PacSurveyData surveyDataTurn;
 teapot->survey(surveyDataTurn,0,teapot->size()-1);
 std::cout <<   "One entire turn     surveyed\n";
 std::cout << "   surveyDataTurn.survey().suml() " << surveyDataTurn.survey().suml() << "\n";
 std::cout << "   these should be near zero:"                                        << "\n";
 std::cout << "      surveyDataTurn.survey().x()    " << surveyDataTurn.survey().x()    << "\n";
 std::cout << "      surveyDataTurn.survey().y()    " << surveyDataTurn.survey().y()    << "\n";
 std::cout << "      surveyDataTurn.survey().z()    " << surveyDataTurn.survey().z()    << "\n";

 return 1;
}
