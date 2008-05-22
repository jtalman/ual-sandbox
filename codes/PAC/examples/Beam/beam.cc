// File        : samples.cc
// Description : These samples illustrate the C++ interface to library
//               Beam.
// Author      : Nikolay Malitsky

#include <iostream>
#include "PAC/Beam/Bunch.hh"

int main()
{

  double energy   = 12.0;       // [GeV]
  double mass     = 0.9382796;  // [GeV]
  double charge   = -1.;

  // ********************************************************
  // Defining a bunch of 1000 particles
  // ********************************************************

  PAC::Bunch bunch(1000);

  // ********************************************************
  // Initialization of beam attributes
  // ********************************************************

  PAC::BeamAttributes ba;

  ba.setEnergy(energy);
  ba.setMass(mass);
  ba.setCharge(charge);

  std::cerr << "energy = " << ba.getEnergy() << "\n"
       << "mass   = " << ba.getMass()   << "\n"
       << "charge = " << ba.getCharge() << "\n";

  bunch.setBeamAttributes(ba);

  // ********************************************************
  // Initialization of particle coordinates
  // ********************************************************

  PAC::Position p;
  for(int i=0; i < bunch.size(); i++) { 
    p.set(i*1.0e-3, 0.0, i*1.0e-3, 0.0, 0.0, 1.0e-5);
    bunch[i].setPosition(p);
  }

  std::cerr << "Let's consider particle #500 \n";
  PAC::Position p500 = bunch[500].getPosition();

  std::cerr << "x  = " << p500.getX()  << "\n"
          "px = " << p500.getPX() << "\n" 
          "y  = " << p500.getY()  << "\n"
          "py = " << p500.getPY() << "\n"
          "ct = " << p500.getCT() << "\n"
          "de = " << p500.getDE() << "\n";       

  // ********************************************************
  // Calculate their moments
  // ********************************************************

  PAC::Position moment;

  for(int order = 0; order < 5; order++){
    moment = bunch.moment(order);

    std::cerr << "x_moment(" << order << ") = " << moment.getX() << "\n";
    std::cerr << "OR \n";
    std::cerr << "x_moment(" << order << ") = " << bunch.moment(0, order) << "\n"; 
  }


}
