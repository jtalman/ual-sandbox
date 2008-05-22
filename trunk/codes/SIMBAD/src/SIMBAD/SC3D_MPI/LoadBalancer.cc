// Library       : SIMBAD
// File          : SIMBAD/SC3D_MPI/LoadBalancer.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include "SIMBAD/SC3D_MPI/LoadBalancer.hh"
#include <mpi.h>

#include <cmath>
#include <cfloat>
#include <cstdlib>

using namespace std;

SIMBAD::LoadBalancer* SIMBAD::LoadBalancer::sTheInstance = 0;

SIMBAD::LoadBalancer::LoadBalancer(PAC::Bunch& bunch)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  localMacs = 0;
  weight = 0.1f;
  paternal.resize(numProcs);
  maternal.resize(numProcs);
  child1.resize(numProcs);
  child2.resize(numProcs);

  nSB = numProcs;  //this is the default value
  macrosInSB.resize(numProcs);

  iterations = 
    (int)(numProcs*nSB*
	  ceil(log(double(numProcs*nSB))));

  chunkSize = 1000;

  ctMinGlobal = DBL_MAX;
  ctMaxGlobal = -DBL_MAX;

  setPartIDs(bunch);

  initPTTable();
}

SIMBAD::LoadBalancer::~LoadBalancer()
{
}

SIMBAD::LoadBalancer& SIMBAD::LoadBalancer::getInstance(PAC::Bunch& bunch)
{
  if(sTheInstance == 0)
    sTheInstance = new SIMBAD::LoadBalancer(bunch);

  return *sTheInstance;
}

void SIMBAD::LoadBalancer::setPartIDs(PAC::Bunch& bunch)
{
  //particle IDs are initialized beginning from 0

  int *localBSArray = new int[numProcs];
  int *globalBSArray = new int[numProcs];

  for(int i = 0; i < numProcs; i++)
    {
      localBSArray[i] = 0;
      globalBSArray[i] = 0;
    }

  localBSArray[myRank] = bunch.size();

  MPI_Allreduce(localBSArray, globalBSArray, numProcs,
		MPI_INT, MPI_SUM, MPI_COMM_WORLD); 

  int runningTotal = 0;
  for(int i = 0; i < myRank; i++)
    runningTotal += localBSArray[i];

  for(int i = 0; i < bunch.size(); i++)
    bunch[i].setId(runningTotal+i);

  delete [] localBSArray;
  delete [] globalBSArray;

  return;
}

void SIMBAD::LoadBalancer::exchangeParticles(PAC::Bunch& bunch)
{
  int gloMacs = 0; //global macros

  localMacs = 0;

  for(int i = 0; i < bunch.size(); i++)
    {
      if(bunch[i].isLost())
	continue;

      localMacs++;
    }

  MPI_Allreduce(&localMacs, &gloMacs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  evolve(gloMacs);

  int i, j, size;
  int position;
  int *recv_size;
  int *send_size;
  //array of char buffers for receiving data
  char **recv_buffer;

  //arrays of MPI_Request structures;
  MPI_Request *send_request;
  MPI_Request *recv_request;
  MPI_Status status;

  send_request = new MPI_Request[numProcs];
  recv_request = new MPI_Request[numProcs];

  recv_buffer = new char*[numProcs];
  recv_size = new int[numProcs];
  send_size = new int[numProcs];

  //initialize
  for(i = 0; i < numProcs; i++)
    {
      recv_buffer[i] = 0;
      recv_size[i] = 0;
      send_size[i] = 0;
    }
    

  clearPTTableEntries();

  bool iterate = true;

  double *startPos = new double[numProcs];
  double *endPos = new double[numProcs];
  int psum = 0;

  double ctMin = DBL_MAX;
  double ctMax = -DBL_MAX;

  for(int i = 0; i < bunch.size(); i++)
    {
      if(bunch[i].isLost())
	continue;

      double ct = bunch[i].getPosition().getCT();

      if(ct < ctMin) ctMin = ct;
      if(ct > ctMax) ctMax = ct;
    }

  ctMinGlobal = DBL_MAX;
  ctMaxGlobal = -DBL_MAX;

  MPI_Allreduce(&ctMin, &ctMinGlobal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ctMax, &ctMaxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  deltaBeam = (ctMaxGlobal-ctMinGlobal)/(float)nSB;

  //fill in startPos and endPos
  for(i = 0; i < numProcs; i++)
    {
      startPos[i] = psum*deltaBeam + ctMinGlobal;
      psum += paternal[i];
      endPos[i] = psum*deltaBeam + ctMinGlobal;
    }

  while(iterate)
    {
      iterate = false;
      for(i = 0; i < bunch.size(); i++)
	{
	  if(bunch[i].isLost())
	    continue;

	  //check if the particle may stay with the current process
	  if(!(bunch[i].getPosition().getCT() >= startPos[myRank] &&
	       bunch[i].getPosition().getCT() <= endPos[myRank]))
	    {


	      //need to transfer the particle
	      for(j = 0; j < numProcs; j++)
		{
		  //locate the transferee process
		  //need to load balance in case the number of processes exceeds the
		  //number of elements
		  if(j != myRank)
		    {
		      //The while is necessary because when a particle
		      //is removed from the herd it is swapped with the
		      //last particle in the herd.  This particle should
		      //then be checked as well.
		      while(bunch[i].getPosition().getCT() >= startPos[j]
			    && bunch[i].getPosition().getCT() <= endPos[j])
			{
			  //j is the process to which the particle belongs
			  //record the the entry in the p_t_table
			  incrementPTTableEntry(myRank, j);
			  
			  //copy the particle to the buffer
			  partToBuffer(bunch.getParticle(i), j);
			  
			  //remove the particle from the herd
			  bunch.erase(i);
			  
			  iterate = true;
			  
			  if(i >= bunch.size())
			    break;
			}
		    }
		}
	    }
	}
    }

  delete [] startPos;
  delete [] endPos;

  startPos = 0;
  endPos = 0;

  syncPTTable();

  for(i = 0; i < numProcs; i++)
    {
      recv_size[i] = retrievePTTableEntry(i, myRank);

      if(recv_size[i] > 0)
	{
	  size = recv_size[i]*getMPSize();
	  recv_buffer[i] = new char[size];

	  MPI_Irecv(recv_buffer[i], size, MPI_PACKED, i, 0, MPI_COMM_WORLD, &(recv_request[i]));

	  //cout << myRank << "  recv from " << i << "  #part = " << recv_size[i] << endl;
	}

    }

  MPI_Barrier(MPI_COMM_WORLD);

  for(i = 0; i < 2*numProcs; i++)
    {
      //We need to arrange that during the communication
      //no process will wait for others to communicate
      if(i%2 == 0)
	{
	  //index range starting with their process number
	  if(myRank%2 == 0)
	    j = myRank + i/2;
	  else
	    j = myRank - i/2;
	}
      else
	{
	  if(myRank%2 == 0)
	    j = myRank - i/2;
	  else
	    j = myRank + i/2;
	}

      if(j >= 0 && j < numProcs)
	{
	  send_size[j] = retrievePTTableEntry(myRank, j);
	  
	  //This is the common case and must be very fast.
	  if(send_size[j] > 0)
	    {
	      //send to process j
	      MPI_Irsend(transBuffer[j].buffer, transBuffer[j].size*getMPSize(), 
			 MPI_PACKED, j, 0, MPI_COMM_WORLD, &(send_request[j]));
	      //cout << myRank << "  sending to " << j << "  #part = " << send_size[j] << endl;
	    }
	}
    }


  for(i = 0; i < numProcs; i++)
    {
      if(send_size[i] > 0)
	{
	  //cout << myRank << " Wait" << endl;
	  MPI_Wait(&(send_request[i]), &status);	  
	}
      
      if(recv_size[i] > 0)
	{
	  //cout << myRank << " Wait" << endl;
	  MPI_Wait(&(recv_request[i]), &status);
	}
    }


  //put the transferred particles in the herd
  //IMPORTANT!!  The particle must be unpacked in the same order as
  //it was packed in partToBuffer.
  for(i = 0; i < numProcs; i++)
    {
      if(recv_buffer[i] != 0)
	{
	  //variable needed in MPI_Unpack
	  position = 0;

	  size = recv_size[i]*getMPSize();
	  for(j = 0; j < recv_size[i]; j++)
	    {
	      int mpIntVal;
	      double mpDoubleVal;
	      PAC::Position insertPosition;
	      PAC::Spin insertSpin;
	      PAC::Particle insertParticle;
	      PAC::Bunch insertBunch(1);
	      
	      MPI_Unpack(recv_buffer[i], size, &position, 
			 &mpIntVal, 1, MPI_INT, MPI_COMM_WORLD);
	      insertParticle.setId(mpIntVal);

	      MPI_Unpack(recv_buffer[i], size, &position, 
			 &mpIntVal, 1, MPI_INT, MPI_COMM_WORLD);
	      insertParticle.setFlag(mpIntVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setX(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setPX(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setY(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setPY(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setCT(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertPosition.setDE(mpDoubleVal);
	      
	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertSpin.setSX(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertSpin.setSY(mpDoubleVal);

	      MPI_Unpack(recv_buffer[i], size, &position,
			 &mpDoubleVal, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	      insertSpin.setSZ(mpDoubleVal);

	      insertParticle.setPosition(insertPosition);
	      insertParticle.setSpin(insertSpin);
	      insertBunch.setParticle(0, insertParticle);
	      bunch.add(insertBunch);
	    }
	}
    }

  delete [] recv_size;
  delete [] send_size;

  delete [] send_request;
  delete [] recv_request;


  //now delete the recv_buffer entries
  for(i = 0; i < numProcs; i++)
    {
      delete [] recv_buffer[i];
      recv_buffer[i] = 0;
    }


  delete [] recv_buffer;

  resetBuffer();

  return;
}

void SIMBAD::LoadBalancer::initPTTable()
{
  /* 
     Example table with 4 processes
     (a,b) = (sender, receiver)

     -------------
     |0,1|0,2|0,3|
     -------------
     |1,0|1,2|1,3|
     -------------
     |2,0|2,1|2,3|
     -------------
     |3,0|3,1|3,2|
     -------------
  */
  int tableSize = numProcs * (numProcs-1);
  int i, j, k, index;

  if(pTTable.size() == 0)
    pTTable.resize(tableSize);
  
  for(i = 0; i < numProcs; i++)
    {
      k = 0;
      for(j = 0; j < numProcs-1; j++)
	{
	  index = i*(numProcs-1)+j;
	  if(i == k)
	    k++;

	  pTTable[index].sender = i;

	  pTTable[index].receiver = k;

	  pTTable[index].n = 0;

	  k++;
	}
    }
  return;
}

void SIMBAD::LoadBalancer::assignMacrosToSB(PAC::Bunch& bunch,
					    vector<vector<int> >& sBIndicesVect)
{
  //ctMinGlobal and deltaBeam have been calculated in exhangeParticles.

  vector<unsigned long int> counters(nSB);

  sBIndicesVect.resize(nSB);

  for(unsigned long int i = 0; i < sBIndicesVect.size(); i++)
    sBIndicesVect[i].resize(0);

  for(int i = 0; i < bunch.size(); i++)
    {
      if(bunch[i].isLost())
	continue;

      //find the sub bunch
      double ct = bunch[i].getPosition().getCT()-ctMinGlobal; //zero scale
      int index = (int)(ct/deltaBeam);

      if(index < 0)
	index = 0;
      else if(index >= nSB)
	index = nSB-1;

      if(sBIndicesVect[index].size() <= counters[index])
	sBIndicesVect[index].resize(sBIndicesVect[index].size()+chunkSize);

      sBIndicesVect[index][counters[index]] = i;
      counters[index]++;      
    }


  //adjust final size to match
  for(unsigned long int i = 0; i < sBIndicesVect.size(); i++)
    sBIndicesVect[i].resize(counters[i]);

  return;
}

void SIMBAD::LoadBalancer::setNSB(int n)
{
  nSB = n;

  macrosInSB.resize(n);

  iterations = 
    (int)(numProcs*nSB*
	  ceil(log(double(numProcs*nSB))));
  return;
}

int SIMBAD::LoadBalancer::getNSB() const
{
  return nSB;
}

int SIMBAD::LoadBalancer::getLocalNSB() const
{
  return localNSB;
}

int SIMBAD::LoadBalancer::getStartSB() const
{
  return startSB;
}

int SIMBAD::LoadBalancer::getNumProcs() const
{
  return numProcs;
}

int SIMBAD::LoadBalancer::getMyRank() const
{
  return myRank;
}

void SIMBAD::LoadBalancer::setIterations(int n)
{
  iterations = n;
  return;
}

int SIMBAD::LoadBalancer::getIterations() const
{
  return iterations;
}

void SIMBAD::LoadBalancer::setWeight(float w)
{
  weight = w;
  return;
}

float SIMBAD::LoadBalancer::getWeight() const
{
  return weight;
}

void SIMBAD::LoadBalancer::setDeltaBeam(float db)
{
  deltaBeam = db;
  return;
}

float SIMBAD::LoadBalancer::getDeltaBeam() const
{
  return deltaBeam;
}

int SIMBAD::LoadBalancer::incrementPTTableEntry(int sender, int receiver)
{
  if(pTTable.size()==0)
    return -1;

  for(unsigned long int i = sender; i < pTTable.size(); i++)
    {
      if(sender == pTTable[i].sender &&
	 receiver == pTTable[i].receiver)
	{
	  pTTable[i].n++;
	  break;
	}

    }

  return 0;
}

int SIMBAD::LoadBalancer::retrievePTTableEntry(int sender, int receiver)
{
  if(pTTable.size()==0)
    return -1;

  for(unsigned long int i = sender; i < pTTable.size(); i++)
    {
      if(sender == pTTable[i].sender &&
	 receiver == pTTable[i].receiver)
	{
	  return pTTable[i].n;
	}
    }

  return 0;
}

void SIMBAD::LoadBalancer::clearPTTableEntries()
{
  for(unsigned long int i = 0; i < pTTable.size(); i++)
    pTTable[i].n = 0;

  return;
}

void SIMBAD::LoadBalancer::syncPTTable()
{
  int i, size;
  int tableSize = pTTable.size();
  int *localTable;
  int *globalTable;
  
  size = (numProcs-1);

  localTable = new int[tableSize];
  globalTable = new int[tableSize];

  //initialize

  for(i = 0; i < tableSize; i++)
    {
      localTable[i] = 0;
      globalTable[i] = 0;
    } 

  for(i = 0; i < size; i++)
    {      
      localTable[size*myRank+i] = pTTable[size*myRank+i].n;
    }

  //for some reason, MPI_Allreduce is WAY FASTER than MPI_Allgather
  MPI_Allreduce(localTable, globalTable, tableSize, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for(i = 0; i < tableSize; i++)
    {
      pTTable[i].n = globalTable[i];
    }

  delete [] localTable;
  delete [] globalTable;

  return;
}



int SIMBAD::LoadBalancer::partToBuffer(PAC::Particle& mp, int processNum)
{
  int i;
  int size;

  //initialize if necessary
  if(!transBuffer)
    {
      transBuffer = (TransferBuffer*)malloc(numProcs*
					    sizeof(TransferBuffer));
      if(!transBuffer)
	return -1;
      
      for(i = 0; i < numProcs; i++)
	{
	  transBuffer[i].buffer = (char*)calloc(chunkSize, getMPSize());
	  if(!transBuffer[i].buffer)
	    return -1;

	  transBuffer[i].size = 0;
	  transBuffer[i].buffSize = chunkSize;
	  transBuffer[i].position = 0;
	}
    }

  //now resize if necessary
  if(transBuffer[processNum].buffSize <= transBuffer[processNum].size)
    {
      size = transBuffer[processNum].buffSize + chunkSize;
      transBuffer[processNum].buffer = 
	(char*)realloc(transBuffer[processNum].buffer, 
		       size*getMPSize());
      if(!transBuffer[processNum].buffer)
	return -1;

      transBuffer[processNum].buffSize = size;
    }


  //add the particle to the buffer
  //total size of the buffer in bytes
  size = transBuffer[processNum].buffSize * getMPSize();
  
  int mpIntVal;
  double mpDoubleVal;

  mpIntVal = mp.getId();
  MPI_Pack(&mpIntVal, 1, MPI_INT, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpIntVal = mp.getFlag();
  MPI_Pack(&mpIntVal, 1, MPI_INT, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getX();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getPX();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getY();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getPY();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getCT();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);

  mpDoubleVal = mp.getPosition().getDE();
  MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	   size, &transBuffer[processNum].position, MPI_COMM_WORLD);


  PAC::Spin* sp = mp.getSpin();

  if(sp)
    {
      mpDoubleVal = mp.getSpin()->getSX();      
      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);
      
      mpDoubleVal = mp.getSpin()->getSY();  
      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);
      
      mpDoubleVal = mp.getSpin()->getSZ();
      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);

    }
  else
    {
      mpDoubleVal = 0;

      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);
      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);
      MPI_Pack(&mpDoubleVal, 1, MPI_DOUBLE, transBuffer[processNum].buffer,
	       size, &transBuffer[processNum].position, MPI_COMM_WORLD);
    }

  transBuffer[processNum].size++;

  return 0;

}


//reset the transBuffer
int SIMBAD::LoadBalancer::resetBuffer()
{
  int i;

  if(!transBuffer)
    return -1;

  //resize and reinitialize
  for(i = 0; i < numProcs; i++)
    {
      if(transBuffer[i].buffSize > chunkSize)
	{
	  transBuffer[i].buffer = 
	    (char*)realloc(transBuffer[i].buffer, 
			   chunkSize * getMPSize());
	  if(!transBuffer[i].buffer)
	    return -1;
	  
	  transBuffer[i].size = 0;
	  transBuffer[i].buffSize = chunkSize;
	  transBuffer[i].position = 0;
	}
      else
	{
	  transBuffer[i].size = 0;
	  transBuffer[i].position = 0;
	}
    }

  return 0;
}

//Returns number of characters in Particle class
//This should really be done in the Particle class itself
//but is implemented here until a better solution is found.
//Any modifications of the Particle Class or any classes
//that are members of Particle must be accounted
//for here (OUCH!).
int SIMBAD::LoadBalancer::getMPSize()
{
  //this accounts for the following members in Particle:
  //int m_id;
  //int m_flag;
  //double m_position.getX()
  //double m_position.getPX()
  //double m_position.getY()
  //double m_position.getPY()
  //double m_position.getCT()
  //double m_position.getDE()
  //double m_spin.getSX()
  //double m_spin.getSY()
  //double m_spin.getSZ()

  return 2*sizeof(int)+9*sizeof(double);
}

void SIMBAD::LoadBalancer::evolve(int gloMacs)
{
  int i, index;
  
  generateStart();
  
  for(i = 0; i < iterations-1; i++)
    {
      mate();
      
      naturalSelection(gloMacs);
      
      mutate();
    }
  
  mate();
  
  naturalSelection(gloMacs);

  index = 0;
  for(i = 0; i < numProcs; i++)
    {
      if(i == myRank)
	startSB = index;

      index += index + paternal[i];
    }

  localNSB = paternal[myRank];
  
  return;
}

void SIMBAD::LoadBalancer::generateStart()
{
  int i;
  int num = nSB/numProcs; //balanced start
  int excess = nSB%numProcs;

  for(i = 0; i < numProcs; i++)
    {
      paternal[i] = num;
      maternal[i] = num;
    }

  //assign excess to nodes starting from 1
  for(i = 1; i <= excess; i++)
    {
      paternal[i]++;
      maternal[i]++;
    }

  return;
}

void SIMBAD::LoadBalancer::mate()
{
  int i;
  int child1_tot = 0;
  int child2_tot = 0;

  //use alternating scheme
  for(i = 0; i < numProcs-1; i++)
    {
      if(i%2 == 0)
	{
	  child1[i] = paternal[i];
	  child2[i] = maternal[i];
	}
      else
	{
	  child1[i] = maternal[i];
	  child2[i] = paternal[i];
	}

      child1_tot += child1[i];
      child2_tot += child2[i];
    }

  //now fit the last number
  child1[numProcs-1] = nSB - child1_tot;
  child2[numProcs-1] = nSB - child2_tot;

  return;
}

void SIMBAD::LoadBalancer::naturalSelection(int gloMacs)
{
  int i;
  int a, b;
  float min;
  float min2 = 0.0f;
  vector<int> &temp1 = paternal; //initialize
  vector<int> &temp2 = paternal; //initialize
  float hvals[4];

  hvals[0] = calcH(gloMacs, 0); //father
  hvals[1] = calcH(gloMacs, 1); //mother
  hvals[2] = calcH(gloMacs, 2); //child1
  hvals[3] = calcH(gloMacs, 3); //child2

  min = hvals[0];
  a = 0;

  for(i = 1; i < 4; i++)
    {
      if(hvals[i] < min)
	{
	  min = hvals[i];
	  a = i;
	}
    }

  for(i = 0; i < 4; i++)
    {
      if(i != a)
	{
	  min2 = hvals[i];
	  break;
	}
    }

  b = 1;
  for(i = 0; i < 4; i++)
    {
      if(i != a && hvals[i] < min2)
	{
	  min2 = hvals[i];
	  b = i;
	}
    }

  //a and b contain the two best values

  switch(a)
    {
    case 0:
      temp1 = paternal;
      break;
    case 1:
      temp1 = maternal;
      break;
    case 2:
      temp1 = child1;
      break;
    case 3:
      temp1 = child2;
      break;
    }

  switch(b)
    {
    case 0:
      temp2 = paternal;
      break;
    case 1:
      temp2 = maternal;
      break;
    case 2:
      temp2 = child1;
      break;
    case 3:
      temp2 = child2;
      break;
    }
      
  //choose the best candidates as new parents
  for(i = 0; i < numProcs; i++)
    {
      paternal[i] = temp1[i];
      maternal[i] = temp2[i];
    }

  return;
}

void SIMBAD::LoadBalancer::mutate()
{
  int i;
  int j;
  int max;

  //mutate maternal

  //pick a proc
  i = rand()%numProcs;

  //inc or dec
  j = rand()%2;

  max = (nSB-(numProcs-1));

  if(j == 0)
    {
      if(maternal[i] == max)
	{
	  maternal[i]--;
	  i++;
	  i = i%numProcs;
	  while(maternal[i] >= max)
	    {
	      i++;
	      i = i%numProcs;
	    }
	  maternal[i]++;
	}
      else
	{
	  maternal[i]++;
	  i++;
	  i = i%numProcs;
	  while(maternal[i] <= 1)
	    {
	      i++;
	      i = i%numProcs;
	    }
	  maternal[i]--;
	}
    }
  else
    {
      if(maternal[i] == 1)
	{
	  maternal[i]++;
	  i++;
	  i = i%numProcs;
	  while(maternal[i] <= 1)
	    {
	      i++;
	      i = i%numProcs;
	    }
	  maternal[i]--;
	}
      else
	{
	  maternal[i]--;
	  i++;
	  i = i%numProcs;
	  while(maternal[i] >= max)
	    {
	      i++;
	      i = i%numProcs;
	    }
	  maternal[i]++;
	}
    }

  //debug
  /*
  int local_sum = 0;
  for(int k = 0; k < numProcs; k++)
    local_sum += maternal[k];
  
  if(local_sum != nElems)
    {
      cerr << local_sum << "  " << nElems  << endl;
      for(int l = 0; l < numProcs; l++)
	cerr << maternal[l] << "  ";
      
      cerr << endl;
      Error::PostFatalError("FATAL ERROR IN LOAD BALANCING");
    }
  */
    return;
}

float SIMBAD::LoadBalancer::calcH(int gloMacs, int type)
{
  int i, j, index;
  int nSubB = 0;  //number of elements for a given process
  float nMacs = 0.0f;  //number of macros for a given process
  float fval = 0.0f;
  float gval = 0.0f;
  float hval = 0.0f;
  float farg = 0.0f;
  float garg = 0.0f;
  vector<int> &member = paternal; //need to initialize reference

  switch(type)
    {
    case 0:
      member = paternal;
      break;
    case 1:
      member = maternal;
      break;
    case 2:
      member = child1;
      break;
    case 3:
      member = child2;
      break;
    }

  index = 0;
  for(i = 0; i < numProcs; i++)
    {
      nSubB = member[i];
      nMacs = 0;

      for(j = 0; j < member[i]; j++)
	{	  
	  nMacs += (float)macrosInSB[index];
	  index++;
	}
      garg = (1.0f - (float)(numProcs * nSubB)/(float)nSB);

      gval += (garg*garg);

      farg = (1.0f - numProcs * nMacs/gloMacs);

      fval += (farg*farg);
    }

  hval = fval + weight * gval;
  return hval;
}
