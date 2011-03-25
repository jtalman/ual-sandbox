
#include "PositionPrinter.h"

PositionPrinter::PositionPrinter()
{
}

void PositionPrinter::open(const char* fileName)
{
    output.open(fileName);
}

void PositionPrinter::close()
{
    output.close();
}

void PositionPrinter::write(int iturn, int ip, PAC::Bunch& bunch)
{
    
      PAC::BeamAttributes& ba = bunch.getBeamAttributes();
      
      double energy = ba.getEnergy();
      double mass   = ba.getMass();
      double G      = ba.getG();
      double gam   = energy/mass;
      double Ggamma = gam*G ; //AUL:29DEC09
      double p     = sqrt(energy*energy - mass*mass);
      double v     = p/gam/energy*UAL::clight;
      double v0byc = p/energy;

      double t0    = ba.getElapsedTime();

      PAC::Position& pos = bunch[ip].getPosition();

      double x  = pos.getX();
      double px = pos.getPX();
      double y  = pos.getY();
      double py = pos.getPY();
      double ct = pos.getCT();
      double de = pos.getDE();

      double wp_time = t0 + (-ct /UAL::clight );
      double ew      = de * p + energy;
      double dew     = de * p;

      double psp0    = get_psp0(pos, v0byc);

      char endLine = '\0';
      char line1[200];
      
      //      sprintf(line1, "%1d %7d    %-15.9e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.10e %-15.10e %c",
      //	      ip, iturn, wp_time, x, px, y, py, ct, de, psp0, ew, endLine);
      //std::cout << "in this output \n";
      
      sprintf(line1, "%1d %7d    %-15.9e %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %c",
	      ip, iturn, wp_time, Ggamma, x, px, y, py, ct, de, endLine);

      output << line1 << std::endl;
}

double PositionPrinter::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);


    return psp0;
}

