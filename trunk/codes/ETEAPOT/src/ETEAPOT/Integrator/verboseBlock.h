//        my fundamental parameter
double gamma0=e0/m0;
//        my fundamental parameter

//        my fundamental constants
double mp         = 1.672621637e-27;                // proton mass (Kg)
//double unitCharge = 1.602176487e-19;                // proton charge (C) http://en.wikipedia.org/wiki/Elementary_charge
  double unitCharge = UAL::elemCharge;                // proton charge (C) http://en.wikipedia.org/wiki/Elementary_charge
double c          = 2.99792458e8;                   // m/s
double GeVperJ    = 1/unitCharge/1e9;
//        my fundamental constants

//      near fundamental parameter
//double v0       = 0.5983790721*c      ;           // m/s    1.79389532e8
  double v0       = c*sqrt(1-1/gamma0/gamma0);
//      near fundamental parameter

//        ca the original maple set
double mcsq       = mp*c*c*GeVperJ;                 // 0.93827231;         // GeV
double G          = 1.7928474;                      //
//double c        = 2.99792458e8;                   // m/s
double g          = 2*G + 2;                        // = 5.5856948
double beta0      = v0/c;                           // 0.5983790721;       //
//double gamma0     = 1/pow(1-beta0*beta0,.5);      // 1.248107349;        //
double Escr0      = gamma0*mcsq         ;           // = 1.171064565 GeV
double K0         = Escr0 - mcsq        ;           // = 0.232792255 GeV
double p0c        = 0.7007405278        ;           // GeV
//double beta0    = 0.5983790721        ;           //
double gamBetSq   = gamma0-1/gamma0     ;
//        ca the original maple set

//        lattice parameters
double R0         = 31.81               ;           // m
double Circ       = 2*PI*R0             ;           // m
double E0         = mp*c*c*gamBetSq/unitCharge/R0;  // (V/m)? Electric field on the central orbit
double p0mks      = gamma0*mp*v0        ;
double p0m        = p0mks*c*GeVperJ     ;
double L0         = mp*v0*gamma0*R0     ;           // R0*p0 kg m^2/s 
//        lattice parameters

//        probe specific values
double gamma      = 0                   ;
double Escr       = 0                   ;           // Escr0 +eV(r) = 1.171064565 + eV(r)
double X          = 0                   ;           // m
double Xsq        = 0                   ;           // m^2
double Y          = 0                   ;           // m
double Ysq        = 0                   ;           // m^2
double Z          = 0                   ;           // m
double Zsq        = 0                   ;           // m^2
double rsq        = 0                   ;           // R0*R0
double r          = 0                   ;           // R0
double v          = 0                   ;           // probe velocity
double fTheta     = 0                   ;
int    iTheta     = 0                   ;
double px         = 0                   ;           // radial cylindrical         particle momentum
double pr         = 0                   ;           // radial cylindrical         particle momentum
double py         = 0                   ;           // up     cylindrical         particle momentum
double pt         = 0                   ;           // forward  cylindrical forward particle momentum
double pz         = 0                   ;           // forward  cylindrical forward particle momentum
double pzsq       = 0                   ;           // forward probe momentum squared
double psq        = 0                   ;           // probe momentum squared
double pp         = 0                   ;           // probe momentum squared
double ptVia_p0   = 0                   ;           // theta  cylindrical forward particle momentum
                                                    //    take p as p0 to infer pt(heta)
double ptVia_Escr0= 0                   ;           // theta  cylindrical forward particle momentum
                                                    //    Use eqn. 8, page 6 to infer pt(heta)
double ptVia_L0   = 0                   ;
                                       // TOO SIMPLE !!!
     X  =R0 + p.getX();
     Xsq=X*X;
     Y  =0 + p.getY();
     Ysq=Y*Y;
     Z  =0;
     Zsq=Z*Z;
     rsq=Xsq+Ysq+Zsq;
     r  =sqrt(rsq);
                                       // TOO SIMPLE !!!

     Escr  = Escr0 + p[5]*p0;
     gamma = (Escr-unitCharge*R0*log(r/R0))/m0;    // V(r) = eR0ln(r/R0)
     v     = c*sqrt(1-1/gamma/gamma);
     px    = p[1]*p0/GeVperJ/c;
     py    = p[3]*p0/GeVperJ/c;
     psq   = (gamma*mp*v)*(gamma*mp*v);
     pp    = sqrt(psq);
     pzsq  = psq-px*px-py*py;
     pz    = sqrt(pzsq);

     iTheta=m_s/Circ;
     fTheta=m_s-iTheta*Circ;
     fTheta=fTheta/R0;

     ptVia_p0   =sqrt(p0*p0-pr*pr-py*py);
       ptVia_Escr0=Escr0-unitCharge*E0*R0*log(r);  // eqn. 8 which includes electric potential
      ptVia_Escr0=ptVia_Escr0*ptVia_Escr0-mcsq*mcsq-pr*pr-py*py;
     ptVia_Escr0=sqrt(ptVia_Escr0);
     ptVia_L0=L0/r;
std::cout
          << " unitCharge         " << unitCharge    << "\n"
          << " Mp (kg)            " << mp            << "\n"
          << " Mp c^2 (J)         " << mp*c*c        << "\n"
          << " gamma0             " << gamma0        << "\n"
          << " v0                 " << v0            << "\n"
          << " mcsq         (GeV) " << mcsq          << "\n"
          << " UAL::pmass   (GeV) " << UAL::pmass    << "\n"
          << " gamma0*mcsq  (GeV) " << gamma0*mcsq   << "\n"
          << " beta0              " << beta0         << "\n"
          << " gamBetSq           " << gamBetSq      << "\n"
          << " Escr0              " << Escr0         << "\n"
          << " E0                 " << E0            << "\n"
          << " beta0              " << beta0         << "\n"
          << " p0mks              " << p0mks         << "\n"
          << " p0mks*c*GeVperJ    " << p0mks*c*GeVperJ << "\n"
          << " p0m                " << p0m             << "\n"
          << " p0 (orig TPOT var) " << p0            << "\n"
          << " e0 (orig TPOT var) " << e0            << "\n"
          << " L0                 " << L0            << "\n"
          << " ___________________" <<                  "\n"
          << " p.getX()           " << p.getX()      << "\n"
          << " p.getPX()          " << p.getPX()     << "\n"
          << " p.getY()           " << p.getY()      << "\n"
          << " p.getPY()          " << p.getPY()     << "\n"
          << " p.getCT()          " << p.getCT()     << "\n"
          << " p.getDE()          " << p.getDE()     << "\n"
          << " r            (m  ) " << r             << "\n"
          << " m_l                " << m_l           << "\n"
          << " m_s                " << m_s           << "\n"
          << " fTheta             " << fTheta        << "\n"
          << " ptVia_Escr0        " << ptVia_Escr0   << "\n"
          << " ptVia_p0           " << ptVia_p0      << "\n"
          << " ptVia_L0           " << ptVia_L0      << "\n"
          << " ptVia_L0*c         " << ptVia_L0*c    << "\n"
          << " ptVia_L0*c*GeVperJ " << ptVia_L0*c*GeVperJ    << "\n"
          << " Escr               " << Escr                  << "\n"
          << " gamma              " << gamma                 << "\n"
          << " v                  " << v                     << "\n"
          << " px                 " << px                    << "\n"
          << " py                 " << py                    << "\n"
          << " psq                " << psq                   << "\n"
          << " pp                 " << pp                    << "\n"
          << " pzsq               " << pzsq                  << "\n"
          << " pz                 " << pz                    << "\n"
          << " Point 0 (Origin)  (" <<  0 << "," <<  0 << "," <<   0 << ")\n"
          << " Point 1 (Entry)   (" <<  X << "," <<  Y << "," << m_s << ")\n"
          << " Point 2 (~V   )   (" << px << "," << py << "," <<  pz << ")\n"
          << " ___________________" <<                  "\n";
