#include "SpinPrinter.h"
#include <math.h>

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
    omega_sum = 0.0;
    omega_num = 0.0;
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

      double Ggamma = gamma*G ; //AUL:29DEC09

      double pz    = get_psp0(pos, v0byc);

      double s2       = sx*sx + sy*sy + sz*sz;
      double p2       = px * px + py * py + pz * pz;
      double spin_g2  = (sx*px+sy*py+sz*pz)/(sqrt(s2)*sqrt(p2));
      double sg2_1    = spin_g2-1.0;

      double wp_time = t0 + (-ct /UAL::clight );

      if(m_vs0 > 0 && spin_g2 < 0) {

          double suml = UAL::OpticsCalculator::getInstance().suml;
          double phase = acos(spin_g2)/2./UAL::pi;

          if(m_turn0 > 0 ) {

              double T0  = (iturn - m_turn0)*suml/v0byc;
              double dT  = -ct + m_ct0;
              double T   = (iturn - m_turn0)*(1 + dT/T0);

              double omega = (1 + phase - m_phase0)/T;

	      omega_sum = omega_sum + (omega - omega0)/omega0;
	      omega_num = omega_num + 1.;
	      double  omega_ave = omega_sum/omega_num;

	      /*
              std::cout.precision(10);

	      std::cout << iturn
			<< "  " << m_phase0<< "  " << phase << "  " << T
			<< ", omega0   = " << omega0
			<< ", omega(s) = " << omega
			<< ", (omega - omega0)/omega0 = " << (omega - omega0)/omega0 
			<< ", omega_ave = " << omega_ave << std::endl;
	      */
          
          }

          m_turn0  = iturn;
          m_phase0 = phase;
          m_ct0    = ct;
      }

      m_vs0 = spin_g2;

      //AULNLD:28JAN10 : calculate spin tune
      double OT00 = SPINK::SpinPropagator::OTs_mat[0][0] ;
      double OT01 = SPINK::SpinPropagator::OTs_mat[0][1] ;
      double OT02 = SPINK::SpinPropagator::OTs_mat[0][2] ;
      double OT10 = SPINK::SpinPropagator::OTs_mat[1][0] ;
      double OT11 = SPINK::SpinPropagator::OTs_mat[1][1] ;
      double OT12 = SPINK::SpinPropagator::OTs_mat[1][2] ;
      double OT20 = SPINK::SpinPropagator::OTs_mat[2][0] ;
      double OT21 = SPINK::SpinPropagator::OTs_mat[2][1] ;
      double OT22 = SPINK::SpinPropagator::OTs_mat[2][2] ;

      double OTsTrace = OT00 +OT11 +OT22 ;

      //std::cout << "stune from acos= " << s_tune << endl;

       //** AUL:13APR10 */
      //double s_tune = atan2((OT01 -OT10),OTsTrace)/(2*3.14159265358979);
      
      double phi = atan2( (OT01-OT10),(OT12-OT21) ) ;
      double cosmu = (OTsTrace-1.)/2 ;
      double theta = atan2( ((OT12+OT21)*sin(phi)),(OT02+OT20) ) ;
      double sinmu = (OT01-OT10)/2./cos(phi) ; 
      double musign = -atan2( sinmu,cosmu ) ;
      //s_tune = mu/(2*3.14159265358979) ;

      /*
     std::cout << "s1_tune= "<< s1_tune << ",  s_tune= " << s_tune << endl ;
      std::cout << OT00 << " " << OT01 << " " << OT02 << endl ;
      std::cout << OT10 << " " << OT11 << " " << OT12 << endl ;
      std::cout << OT20 << " " << OT21 << " " << OT22 << endl ;
      */
      
      double s_tune = acos((OTsTrace-1.)/2.)/(2*3.14159265358979);

      if(musign <= 0.){s_tune = 1.-s_tune;}

      //std::cout << "sin(mu)= " << sinmu << ", cos(mu)= " << cosmu << endl;
      //std::cout << "stune from atan= " << s_tune << endl;

      //std::cout << "stune = " << s_tune << endl;

      //AULNLD:28JAN10-----------

      // double phase   = cos(omega0*iturn*2.0*UAL::pi); // acos(spin_g2) - m_phase0;     

      char endLine = '\0';
      char line2[200];

      //      sprintf(line2, "%1d %7d    %-15.9e %-16.7e %-16.7e %-16.7e %-16.7e %c",
      //              ip, iturn, sx, sy, sz, s, spin_g2, endLine);

      /*AUL:29DEC09
      sprintf(line2, "%1d %7d    %-18.9e %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %c",
	      ip, iturn, wp_time, sx, sy, sz, spin_g2, sg2_1, endLine);
      */
      //AULNLD:28JAN10
      sprintf(line2, "%1d %7d  %-18.9e % -16.7e % -16.7e % -16.7e %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %c",
	      ip, iturn, wp_time, gamma, Ggamma, sx, sy, sz, spin_g2, sg2_1, s_tune, endLine);

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
