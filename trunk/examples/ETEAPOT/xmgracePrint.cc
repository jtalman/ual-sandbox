#include "xmgracePrint.hh"

xmgracePrint::xmgracePrint()
{
}

void xmgracePrint::open(const char* fileName)
{
    output.open(fileName);
}

void xmgracePrint::close()
{
    output.close();
}

void xmgracePrint::write(int iturn, int ip, PAC::Bunch& bunch)
{
    
      PAC::BeamAttributes& ba = bunch.getBeamAttributes();
      
      double energy = ba.getEnergy();
      double mass   = ba.getMass();

      double gam   = energy/mass;
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

      double psp0    = get_psp0(pos, v0byc);

      char endLine = '\0';
      char line1[200];
      
      sprintf(line1, "%7d %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %c",
	      iturn, x, px, y, py, ct, de, endLine);

      output << line1 << std::endl;
}

double xmgracePrint::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);


    return psp0;
}

