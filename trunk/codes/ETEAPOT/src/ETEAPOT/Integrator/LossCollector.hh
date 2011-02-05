// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/LossCollector.hh


#ifndef ETEAPOT_LOSS_COLLECTOR_HH
#define ETEAPOT_LOSS_COLLECTOR_HH

#include <string>
#include "UAL/Common/Def.hh"
#include "PAC/Beam/Position.hh"
#include "PAC/Beam/Bunch.hh"

namespace ETEAPOT {

  class LossCollector{
  private:
    
    void DeleteArrays();
   
    int sentry;
    int flag;  //this flag is for the error that the collector is full!

  protected:
    int Nparticles;  //totla number of particles in the registerd bunch
    int *p_index;  //contains particle index
    int *turns;     // turn number of loss
    PAC::Position *positions;  //particle position
    int *elem_index;           //element index
    float *elem_location;      //element s coordinate
    int turn;   //internal turn counter.
    std::string *name; //names

    static LossCollector* s_theInstance;

  public: 
    LossCollector();
    ~LossCollector();

    static LossCollector& GetInstance();

    void Clear();

    const int GetNLost() const {return sentry;};
    const int GetParticleIndex(int i)const {return (i<sentry) ? p_index[i] : -1;}
    const int GetTurn(int i)const {return (i<sentry) ? turns[i] : -1;}
    const PAC::Position GetPosition(int i) const {return (i<sentry) ? positions[i] :PAC::Position();}
    const int GetElemIndex(int i) const {return (i<sentry) ? elem_index[i] : -1;}
    const float GetLocation(int i) const {return (i<sentry) ? elem_location[i] : -1;}
    const std::string GetElemName(int i) const {return (i<sentry) ? name[i] : "";}


    void RegisterBunch(const PAC::Bunch &b);
    void RegisterLoss(int pind, PAC::Position &pos, int eind, float s, std::string n); 
    //                  particle index,position, element index, s coordinate, name 
    void SetTurn(int t){turn =t;}
    int IsFull(){return (sentry==Nparticles)? 1 : 0;} //returns 1 if Lost Collector is full
    void Write(const char* filename);
  };

}


#endif
