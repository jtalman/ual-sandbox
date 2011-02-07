// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/LostCollector.hh


#include <fstream>
#include <iostream>
#include <iomanip>
#include "LossCollector.hh"

#define  NO_PARTICLE -1 
// this flag says that there is no particle at p_index[?]

using namespace std;

ETEAPOT::LossCollector* ETEAPOT::LossCollector::s_theInstance=0;

ETEAPOT::LossCollector::LossCollector()
{
  Nparticles=0;
  p_index=NULL;
  turns=NULL;
  positions=NULL;
  elem_index=NULL;
  elem_location=NULL;
  sentry=-1;
  flag=0;
}

ETEAPOT::LossCollector::~LossCollector()
{ 
  DeleteArrays();
}


ETEAPOT::LossCollector& ETEAPOT::LossCollector::GetInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new ETEAPOT::LossCollector();
  }
  return *s_theInstance;
}

void ETEAPOT::LossCollector::DeleteArrays()
{

  if (Nparticles){
    delete [] p_index;
    delete [] turns;
    delete [] positions;
    delete [] elem_index;
    delete [] elem_location;
    delete [] name;
  }
  Nparticles=0;
  sentry=0;
}

void ETEAPOT::LossCollector::Clear()
{
  int i;

  for(i=0;i<Nparticles;i++){
    p_index[i]=NO_PARTICLE;
    turns[i]=elem_index[i]=0;
    positions[i].set(0,0,0,0,0,0);
    elem_location[i]=0.0;
    name[i]="";
  }
  sentry=0;
  flag=1;
}



void ETEAPOT::LossCollector::RegisterBunch(const PAC::Bunch &b)
{
  if(b.size()!= Nparticles){
    DeleteArrays();
    Nparticles=b.size();
    p_index=new int[Nparticles];
    turns=new int[Nparticles];
    positions=new PAC::Position[Nparticles];
    elem_index=new int[Nparticles];
    elem_location=new float[Nparticles];
    name=new string[Nparticles];
  }
  Clear();
  sentry=0;

}

void ETEAPOT::LossCollector::RegisterLoss(int i, PAC::Position &pos, int eind, float s, std::string n)
  //void ETEAPOT::LostCollector::RegisterLoss(int i, PAC::Position &pos, int eind)
{
  int j;

  cout<<"There is a loss! Particle "<<i<<" at position "<<s<<" in turn "<<turn<<endl;
  // cout<<"There is a loss! Particle "<<i<<" at position "<<s<< endl;
  if (Nparticles && sentry<Nparticles){
    p_index[sentry]=i;
    for(j=0;j<6;j++) positions[sentry][j]=pos[j];
    turns[sentry]=turn;
    elem_index[sentry]=eind;
    elem_location[sentry]=s;
    name[sentry]=n;
    sentry++;
  }
    
  if(sentry>=Nparticles && flag) {
    cerr<<"LostCollector full on turn "<<turn<<endl;
    flag=0;
  }
}
      


void ETEAPOT::LossCollector::Write(const char* filename)
{
  ofstream file(filename);
  int i=0;
  int j;

  if(!file) {
    cerr << "Cannot open LostCollector output file " << filename <<endl;
    return;
  }

  // file<<"Particle  Turn  Location        s               x               x'               y              y'               -ds              dp"<<endl;
  file<<"Particle  Turn  Element[ Index     s    Name  ]            x                x'                  y                 y'                   ds                dp"<<endl;
  file.fill(' ');
   file.setf(ios::left  | ios::scientific);
  cout<<"There are "<<sentry<<" lost particles."<<endl;
  
  for(i=0;i<sentry;i++){   
    file.width(11);
    file<<p_index[i];
    file.width(7);
    file<<turns[i];
    file.width(6);
    file<<elem_index[i];
    file<<setprecision(7);
    file<<setw(15)<<elem_location[i];
    file<<name[i]<<setw(10-name[i].size())<<" ";
    file<<setprecision(10);
    for(j=0;j<6;j++){
      file.width(19);
      file<<positions[i][j];
    }
    file<<endl;
   
  }
  file.close();

}
