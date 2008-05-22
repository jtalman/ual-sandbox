
#include "BETACOOL/Ring.hh"
#include "BETACOOL/CompositeTracker.hh"

BETACOOL::CompositeTracker::CompositeTracker()
{
}

void BETACOOL::CompositeTracker::registerEffect(const char* effect, const char* elname)
{
  PacLattice& lattice = BETACOOL::Ring::getInstance().getLattice();
  int index = -1;
  for(int i = 0; i < lattice.size(); i++){
    if(!lattice[i].getDesignName().compare(elname)) {
      index = i;
      break;
    }
  }

  if(index < 0) {
    std::cout << "BETACOOL::CompositeTracker::register: there is NO " << elname << std::endl;
  } else {
    xEffect* Effect = xEffect::GetEffect(effect);
    if (Effect)
    {
      Effect->Lattice = iRing[index].Lattice;
      Effect->Use = true;
      std::cout << effect << " reset by " << elname << " with betax = " << Effect->Lattice.betax() << ", betay = " << Effect->Lattice.betay() << std::endl;
    }
  }
}

void BETACOOL::CompositeTracker::setTimeStep(double timeStep)
{
  iTime.TimeStep(doubleU(timeStep, s_));
}

void BETACOOL::CompositeTracker::propagate(PAC::Bunch& bunch)
{
  readBunch(bunch);

  iBeam.Emit = iBeam.Get_Emit(iRing.LATTICE);

  iDynamics.Drawing(iRing.LATTICE);

  //calculateHistogram(iRing.Llattice);

  xLattice Lattice2 = iRing.LATTICE;
  for (int i = 0; i < xEffect::ACount; i++)
  {
    if (xEffect::AItems[i]->Use)
    {
      transRotate(Lattice2, xEffect::AItems[i]->Lattice);
      addKick(i);
      Lattice2 = xEffect::AItems[i]->Lattice;
    }
  }

  transRotate(Lattice2, iRing.LATTICE);
  longRotate();

  ++iTime;

  writeBunch(bunch);
}

void BETACOOL::CompositeTracker::setLattice(PacTwissData& twiss)
{
  iRing.LATTICE.betax = twiss.beta(0);
  iRing.LATTICE.betay = twiss.beta(1);
  iRing.LATTICE.alfax = twiss.alpha(0);
  iRing.LATTICE.alfay = twiss.alpha(1);
  iRing.LATTICE.Dx    = twiss.d(0);
  iRing.LATTICE.Dy    = twiss.d(1);
  iRing.LATTICE.Dpx   = twiss.dp(0);
  iRing.LATTICE.Dpy   = twiss.dp(1);
}

void BETACOOL::CompositeTracker::readBunch(PAC::Bunch& bunch)
{
  int np = bunch.size();
  iBeam.Number(np);  // set size of beam array only when bunch.size != beam.size

  int num = 0;
  for(int ip =0; ip < np; ip++)
  {
    if (bunch[ip].isLost()) continue;
    PAC::Position& pos = bunch[ip].getPosition();
    iBeam[num][0] = pos.getX();
    iBeam[num][1] = pos.getPX();
    iBeam[num][2] = pos.getY();
    iBeam[num][3] = pos.getPY();
    iBeam[num][4] = pos.getCT();
    iBeam[num][5] = pos.getDE();
    num++;
  }
  iBeam.Number(num);     // set number of using particles

  for (int j2 = 0; j2 < iBeam.Number(); j2++)
    iBeam.MinusDisp(iRing.LATTICE, j2);

}

void BETACOOL::CompositeTracker::writeBunch(PAC::Bunch& bunch)
{

  for (int j2 = 0; j2 < iBeam.Number(); j2++)
    iBeam.PlusDisp(iRing.LATTICE, j2);

  for (int ip =0; ip < iBeam.Number(); ip++)
  {
    bunch[ip].setFlag(0);  // set to alive
    PAC::Position& pos = bunch[ip].getPosition();
    pos.set(iBeam[ip][0], iBeam[ip][1], iBeam[ip][2],
            iBeam[ip][3], iBeam[ip][4], iBeam[ip][5]);
  }

  for (int j = iBeam.Number(); j < bunch.size(); j++)
     bunch[j].setFlag(1); // set to lost

}

void BETACOOL::CompositeTracker::calculateHistogram(xLattice& Lattice)
{
   if (iLosses.Use && (iBeam.GenerateOn == 1))
   {
      iBeam.EmitSort(100, Lattice);
      for (int i = 0; i < 3; i++)
         iBeam.Hystogram(iBeam.Hyst3[i], iBeam.Division, i);
   }
}

void BETACOOL::CompositeTracker::transRotate(xLattice& Lattice1, xLattice& Lattice2)
{

  Lattice1.mux = M_PI * (1. + xDistributor::Flatten());
  Lattice1.muy = M_PI * (1. + xDistributor::Flatten());

  matrixU Matrix(4,4);
  Matrix = iRing.CalcTransformMatrix(Lattice1, Lattice2);

  for (int j0 = 0; j0 < iBeam.Number(); j0++)
  {
    iBeam.Matrix_2x2(j0, Matrix);
  }
}

void BETACOOL::CompositeTracker::longRotate()
{
  double mu  = M_PI * (1. + xDistributor::Flatten());

  if(iBeam.bunched)
  for (int j4 = 0; j4 < iBeam.Number(); j4++)
  {  iBeam[j4][4] = cos(mu)*iBeam[j4][4] + xLattice::B_s() * sin(mu)*iBeam[j4][5];
     iBeam[j4][5] = cos(mu)*iBeam[j4][5] -
       (((iBeam[j4][4]-xLattice::B_s() * sin(mu)*iBeam[j4][5])/cos(mu))/xLattice::B_s())*sin(mu);
   }
}


void BETACOOL::CompositeTracker::addKick(int i){

  for (int j1 = 0; j1 < iBeam.Number(); j1++)
    iBeam.PlusDisp (xEffect::AItems[i]->Lattice, j1);

  xEffect::AItems[i]->Kick(iTime, iBeam, iRing);

  for (int j2 = 0; j2 < iBeam.Number(); j2++)
    iBeam.MinusDisp(xEffect::AItems[i]->Lattice, j2);
}
