
#include "SpinPrinter.h"

SpinPrinter::SpinPrinter()
{
}

void SpinPrinter::open(const char* fileName)
{
    output.open(fileName);
}

void SpinPrinter::close()
{
    output.close();
}

void SpinPrinter::write(int iturn, int ip, PAC::Bunch& bunch)
{
    
      PAC::BeamAttributes& ba = bunch.getBeamAttributes();

      double t0    = ba.getElapsedTime();

      double sx = bunch[ip].getSpin()->getSX();
      double sy = bunch[ip].getSpin()->getSY();
      double sz = bunch[ip].getSpin()->getSZ();

      PAC::Position& pos = bunch[ip].getPosition();

      double x  = pos.getX();
      double px = pos.getPX();
      double y  = pos.getY();
      double py = pos.getPY();
      double ct = pos.getCT();
      double de = pos.getDE();

      double wp_time = t0 + (-ct /UAL::clight );

      double spin_g2 = (sx*px+sy*py+sz*(1.0+x/100.0))/sqrt(sx*sx+sy*sy+sz*sz)/sqrt(px*px+py*py+(1.0+x/100.0)*(1.0+x/100.0));

      char endLine = '\0';
      char line2[200];

      sprintf(line2, "%1d %7d    %-15.9e %-16.7e %-16.7e %-16.7e %-16.7e %c",
	      ip, iturn, wp_time, sx, sy, sz, spin_g2, endLine);

      output << line2 << std::endl;
}



