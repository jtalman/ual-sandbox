// Library     : PAC
// File        : PAC/Optics/LinearMapper.cc
// Copyright   : see Copyright file

#include "PAC/Beam/Bunch.hh"
#include "Optics/LinearMapper.hh"

PAC::LinearMapper::LinearMapper()
{
  init();
}

PAC::LinearMapper::LinearMapper(const PAC::LinearMapper& rhs)
{
  init(rhs);
}

PAC::LinearMapper::~LinearMapper()
{
}

/*
PAC::Algorithm* PAC::LinearMapper::clone()
{
  return new LinearMapper(*this);
}
*/
void PAC::LinearMapper::setMap(const PacVTps& vtps)
{
  a10 = vtps(0, 0); a11 = vtps(0, 1); a12 = vtps(0, 2); a13 = vtps(0, 3); a14 = vtps(0, 4); a15 = vtps(0, 5); a16 = vtps(0, 6);
  a20 = vtps(1, 0); a21 = vtps(1, 1); a22 = vtps(1, 2); a23 = vtps(1, 3); a24 = vtps(1, 4); a25 = vtps(1, 5); a26 = vtps(1, 6);
  a30 = vtps(2, 0); a31 = vtps(2, 1); a32 = vtps(2, 2); a33 = vtps(2, 3); a34 = vtps(2, 4); a35 = vtps(2, 5); a36 = vtps(2, 6);  
  a40 = vtps(3, 0); a41 = vtps(3, 1); a42 = vtps(3, 2); a43 = vtps(3, 3); a44 = vtps(3, 4); a45 = vtps(3, 5); a46 = vtps(3, 6); 
  a50 = vtps(4, 0); a51 = vtps(4, 1); a52 = vtps(4, 2); a53 = vtps(4, 3); a54 = vtps(4, 4); a55 = vtps(4, 5); a56 = vtps(4, 6);   
  a60 = vtps(5, 0); a61 = vtps(5, 1); a62 = vtps(5, 2); a63 = vtps(5, 3); a64 = vtps(5, 4); a65 = vtps(5, 5); a66 = vtps(5, 6);    
}

void PAC::LinearMapper::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = dynamic_cast<PAC::Bunch&>(probe);

  double x0, px0, y0, py0, ct0, de0;
  double x, px, y, py, ct, de;

  int size = bunch.size();
  for(int i=0; i < size; i++){

    if(bunch[i].getFlag() > 0 ) continue;

    PAC::Position& pos = bunch[i].getPosition();
    
    x0  = pos.getX();
    px0 = pos.getPX();
    y0  = pos.getY();
    py0 = pos.getPY();
    ct0 = pos.getCT();
    de0 = pos.getDE();
        
    x  = a10 + a11*x0 + a12*px0 + a13*y0 + a14*py0 + a15*ct0 + a16*de0;
    px = a20 + a21*x0 + a22*px0 + a23*y0 + a24*py0 + a25*ct0 + a26*de0;
    y  = a30 + a31*x0 + a32*px0 + a33*y0 + a34*py0 + a35*ct0 + a36*de0;
    py = a40 + a41*x0 + a42*px0 + a43*y0 + a44*py0 + a45*ct0 + a46*de0;
    ct = a50 + a51*x0 + a52*px0 + a53*y0 + a54*py0 + a55*ct0 + a56*de0;
    de = a60 + a61*x0 + a62*px0 + a63*y0 + a64*py0 + a65*ct0 + a66*de0;

    pos.set(x, px, y, py, ct, de);
    
  }

}

void PAC::LinearMapper::init()
{
  a10 = a11 = a12 = a13 = a14 = a15 = a16 = 0.0;
  a20 = a21 = a22 = a23 = a24 = a25 = a26 = 0.0;
  a30 = a31 = a32 = a33 = a34 = a35 = a36 = 0.0;  
  a40 = a41 = a42 = a43 = a44 = a45 = a46 = 0.0; 
  a50 = a51 = a52 = a53 = a54 = a55 = a56 = 0.0;   
  a60 = a61 = a62 = a63 = a64 = a65 = a66 = 0.0;  
}

void PAC::LinearMapper::init(const PAC::LinearMapper& rhs)
{
  a10 = rhs.a10; a11 = rhs.a11; a12 = rhs.a12; a13 = rhs.a13; a14 = rhs.a14; a15 = rhs.a15; a16 = rhs.a16;
  a20 = rhs.a20; a21 = rhs.a21; a22 = rhs.a22; a23 = rhs.a23; a24 = rhs.a24; a25 = rhs.a25; a26 = rhs.a26;
  a30 = rhs.a30; a31 = rhs.a31; a32 = rhs.a32; a33 = rhs.a33; a34 = rhs.a34; a35 = rhs.a35; a36 = rhs.a36;  
  a40 = rhs.a40; a41 = rhs.a41; a42 = rhs.a42; a43 = rhs.a43; a44 = rhs.a44; a45 = rhs.a45; a46 = rhs.a46; 
  a50 = rhs.a50; a51 = rhs.a51; a52 = rhs.a52; a53 = rhs.a53; a54 = rhs.a54; a55 = rhs.a55; a56 = rhs.a56;   
  a60 = rhs.a60; a61 = rhs.a61; a62 = rhs.a62; a63 = rhs.a63; a64 = rhs.a64; a65 = rhs.a65; a66 = rhs.a66;  
}


