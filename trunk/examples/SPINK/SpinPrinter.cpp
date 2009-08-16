
#include "SpinPrinter.h"

SpinPrinter::SpinPrinter()
{
}

void SpinPrinter::setLength(double suml)
{
}

void SpinPrinter::open(const char* fileName)
{
    output.open(fileName);
    m_vs0 = 0.0;

    m_turn0  = 0.0;
    m_phase0 = 0.0;
    m_ct0    = 0.0;
}

void SpinPrinter::close()
{
    output.close();
}

void SpinPrinter::write(int iturn, int ip, PAC::Bunch& bunch)
{
    
      PAC::BeamAttributes& ba = bunch.getBeamAttributes();

      double energy = ba.getEnergy();
      double mass   = ba.getMass();
      double G      = ba.getG();

      double gamma  = energy/mass;

      double p     = sqrt(energy*energy - mass*mass);
      double v0byc = p/energy;

      double t0    = ba.getElapsedTime();

      // spinye

      double sx = bunch[ip].getSpin()->getSX();
      double sy = bunch[ip].getSpin()->getSY();
      double sz = bunch[ip].getSpin()->getSZ();

      double s = sqrt(sx*sx + sy*sy + sz*sz);

      // position

      PAC::Position& pos = bunch[ip].getPosition();

      double px = pos.getPX();
      double py = pos.getPY();
      double ct = pos.getCT();
      
      // output

      double omega0 = gamma*G;

      double pz    = get_psp0(pos, v0byc);

      double spin_g2 = sx*px+sy*py+sz*pz;

      if(m_vs0 > 0 && spin_g2 < 0) {

          double suml = UAL::OpticsCalculator::getInstance().suml;
          double phase = acos(spin_g2)/2./UAL::pi;

          if(m_turn0 > 0 ) {

              double T0  = (iturn - m_turn0)*suml/v0byc;
              double dT  = -ct + m_ct0;
              double T   = (iturn - m_turn0)*(1 + dT/T0);

              double omega = (1 + phase - m_phase0)/T;

              std::cout.precision(10);
          
              std::cout << iturn
                     << ", omega(s) = " << omega
                     << ", (omega - omega0)/omega0 = " << (omega - omega0)/omega0 << std::endl;
          }

          m_turn0  = iturn;
          m_phase0 = phase;
          m_ct0    = ct;
      }

      m_vs0 = spin_g2;

      // double phase   = cos(omega0*iturn*2.0*UAL::pi); // acos(spin_g2) - m_phase0;     

      char endLine = '\0';
      char line2[200];

      sprintf(line2, "%1d %7d    %-15.9e %-16.7e %-16.7e %-16.7e %-16.7e %c",
	      ip, iturn, sx, sy, sz, s, spin_g2, endLine);

      output << line2 << std::endl;
}

void  SpinPrinter::calculateOmega()
{

  
}

double SpinPrinter::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);


    return psp0;
}




