// Library       : SIMBAD
// File          : SIMBAD/SC3D_MPI/LoadBalancer.hh
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#ifndef UAL_SIMBAD_LOADBALANCER_HH
#define UAL_SIMBAD_LOADBALANCER_HH

#include "PAC/Beam/Bunch.hh"
#include "PAC/Beam/Particle.hh"
#include <vector>

namespace SIMBAD {

  /** Load Balance class for 3D space charge calculation */

//structure used to transfer particles for Parallel running
  class TransferBuffer{
  public:
    //position information for MPI_PACK
    int position;
    
    //current number of elements in buffer
    int size;
    
    //current size of the buffer in elements
    int buffSize;
    char *buffer;
  };

  class PTransTableEntry{
  public:
    int sender; //sender's id
    int receiver; //reciever's id
    int n; //number of particles to send
  };

  class LoadBalancer{
  public:
    LoadBalancer(PAC::Bunch& bunch);
    ~LoadBalancer();
    /** Returns a singleton */
    static LoadBalancer& getInstance(PAC::Bunch& bunch);

    void setPartIDs(PAC::Bunch& bunch);
    void exchangeParticles(PAC::Bunch& bunch);
    void assignMacrosToSB(PAC::Bunch& bunch,
			  vector<vector<int> > &sBIndicesVect);

    void setNSB(int n);
    int getNSB() const;
    int getLocalNSB() const;
    int getStartSB() const;
    int getNumProcs() const;
    int getMyRank() const;
    void setIterations(int n);
    int getIterations() const;
    void setWeight(float w);
    float getWeight() const;
    void setDeltaBeam(float db);
    float getDeltaBeam() const;
    
  private:
    int nSB;  //user parameter
    int localNSB;
    int startSB;
    int numProcs;
    int myRank;
    int iterations;
    int localMacs;  //local number of macros
    int chunkSize;  //for memory management of buffers
    vector<int> macrosInSB;
    vector<int> paternal;
    vector<int> maternal;
    vector<int> child1;
    vector<int> child2;
    vector<PTransTableEntry> pTTable;
    TransferBuffer *transBuffer;
  
    double ctMinGlobal;
    double ctMaxGlobal;
  
    float weight;
    double deltaBeam; //possible user param

    int incrementPTTableEntry(int sndr, int rcvr);
    int retrievePTTableEntry(int sndr, int rcvr);
    void initPTTable();
    void clearPTTableEntries();
    void syncPTTable();
    void evolve(int gloMacs);
    void generateStart();
    void mate();
    void naturalSelection(int gloMacs);
    void mutate();
    float calcH(int gloMacs, int type);

    //move Particle to buffer for transfer to process processNum
    int partToBuffer(PAC::Particle &mp, int processNum);
    
    //reset the transBuffer
    int resetBuffer();

    int getMPSize();

    static LoadBalancer* sTheInstance;
  };
};

#endif
