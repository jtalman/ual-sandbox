
#include <iostream>
#include "TEAPOT/StringSC/StringSCSolver.hh"
#include <fstream>
#include <assert.h>

extern std::ofstream forcecomp;
// extern std::ofstream moments;

TEAPOT::StringSCSolver* TEAPOT::StringSCSolver::s_theInstance = 0;

TEAPOT::StringSCSolver& TEAPOT::StringSCSolver::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new TEAPOT::StringSCSolver();
  }
  return *s_theInstance;
}

TEAPOT::StringSCSolver::StringSCSolver()
{
  // relative error
  m_epsrel     = 1.0e-4;
  m_frac_strL = 0.001;

  // maximum number of iterations
  m_maxIters   = 100;  

  // root bracketing algorithm
  p_fRootSolver  = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

  // root finding algorithm using derivatives
  p_fdfRootSolver = gsl_root_fdfsolver_alloc(gsl_root_fdfsolver_steffenson); 

  m_counter = 0;

  // string length
  m_strL = 0.0;

  // m_tinyByRi = 2.99e-7;
  m_tinyByRi = 1.0e-10;

  m_bendfac = 1;
}

TEAPOT::StringSCSolver::~StringSCSolver()
{
  gsl_root_fsolver_free(p_fRootSolver);
  gsl_root_fdfsolver_free(p_fdfRootSolver);
}

void TEAPOT::StringSCSolver::setCounter(int counter)
{
  m_counter = counter;
}

// 5 July 2007. The occasional close encounter still give unacceptably
// large emittance growth.  This causes an unphysically large dependence
// on vertical beam height. (See M. Bassetti, "Analytical Formula for the 
// Centrifugal Space Charge Effects", CERN/LEP-TH/86-13.) 

// When evaluating string forces, "y" occurs only in the form "y*y",
// (except as a multiplicative factor in calculating vertical force 
// components); and the beam height is always much wider than high.
// By assigning the string a height, e.g. "height=yhW/10", we soften
// close encounters, while still preserving the dependence of density
// on "y" which may cause emittance growth. We therefore make the replacement
//      y*y -> y*y+(yhW/10)*(yhW/10)
// in the denominators of the integrands that give string forces.
// In effect this gives every string an effective height equal to 
// about twice "yhW/10". Note the following averaging;

//                                  / y0
//   /        1          \       1 |         dytw            1         1                1
//  < ------------------  >  =  -- |  ------------------- = --- ----------------- > ------------------
//   \(ytw^2 + a^2)^(3/2)/      y0 |  (ytw^2 + a^2)^(3/2)   a^2 (y0^2 + a^2)^(1/2)  (y0^2 + a^2)^(3/2)
//                                / 0 
//
// The final term has the same form as the integrand factors in
// the formulas for the forces due to a string of vanishing height.
// Rather than integrating analytically over "y", we simply add
// the term "strH^2" to the denominators of the force terms.
// This alteration has negligible effect for all but close encounters,
// and, for those, the kick is less than the kick that would be
// obtained by integrating accurately over the string height, and
// MUCH less than the maximum kick that could result if the string
// height were zero.

void TEAPOT::StringSCSolver::setStringH(double strH)
{
  m_strH = strH;
}

void TEAPOT::StringSCSolver::setStringL(double strL)
{
  m_strL = strL;
}

void TEAPOT::StringSCSolver::setMaxIters(int maxIters)
{
  m_maxIters = maxIters;
}

void TEAPOT::StringSCSolver::setMaxBunchSize(int np)
{
  m_force.resize(np);
  for(int ip=0; ip < np; ip++){
    m_force[ip].resize(np);
  }
}

void TEAPOT::StringSCSolver::propagate(PAC::Bunch& bunch)
{
  propagate(bunch, m_Ri, m_sRi, m_L);
}

    /* 
    3 June, 2006. Inverse bend radius Ri is retained as positive value Ri=|Ri|
    and the bend orientation is maintained by (sign) sRi. Note though that 
    sRi=1 when Ri=0. Bend angles, with their correct algebraic signs, are entered  
    via instructions such as 
              scSolver.setAngle("k2", -0.0163625);
    in "main.cc".
    */

void TEAPOT::StringSCSolver::propagate(PAC::Bunch& bunch, double Ri, int sRi, double L)
{
  // std::cout << "Ri: " << Ri << "  sRi:  " << sRi << "\n"  << std::endl;

  for(int is = 0; is < bunch.size(); is++){
    for(int it = 0; it < bunch.size(); it++){ 
      m_force[is][it].fx = 0.0;
      m_force[is][it].fy = 0.0;
      m_force[is][it].fs = 0.0;
    }
  }

  calculateForce(bunch, Ri, sRi);
  propagate(bunch, L);    // L is length of bending element
  m_counter++;
}

void TEAPOT::StringSCSolver::calculateForce(PAC::Bunch& bunch, double Ri, int sRi)
{
  for(int is = 0; is < bunch.size(); is++){
    if(bunch[is].isLost()) continue;
    for(int it = 0; it < bunch.size(); it++){ 
      if(bunch[it].isLost()) continue;
      calculateForce(bunch, Ri, sRi, is, it);
    }
  }

  // The following code should be commented out unless the longitudinal
  // variation of bunch length "sig_ct" and energy spread "sig_de" is required.

  // /*
  double ct;
  double ct_ave = 0;
  double ct2_ave = 0;
  double de;
  double de_ave = 0;
  double de2_ave = 0;

  PAC::Position p = bunch[0].getPosition();
  int Np = bunch.size();
  double e = bunch.getBeamAttributes().getEnergy();
 
  for(int it = 0; it < Np; it++){
    p = bunch[it].getPosition();
    ct = p.getCT();
    ct_ave += ct/Np;
    ct2_ave += ct*ct/Np;

    de = p.getDE();
    de_ave += de/Np;
    de2_ave += de*de/Np;
  }
  double sig_ct = 1000*sqrt( ct2_ave - ct_ave*ct_ave );
  double sig_de = 1000*e*sqrt( de2_ave - de_ave*de_ave );
  // moments << m_counter << "\t" << sig_ct << "\t" << sig_de << std::endl;
  // */
}

void TEAPOT::StringSCSolver::calculateForce(PAC::Bunch& bunch, double Ri, int sRi, int is, int it)
{

  double e = bunch.getBeamAttributes().getEnergy();
  double m = bunch.getBeamAttributes().getMass();  // m = 0.000511034 
  double gamma = e/m;
  double g2i = 1.0/gamma/gamma;
  double p_0 = sqrt(e*e-m*m);        // momentum and energy units are GeV
  double beta = sqrt(gamma*gamma - 1)/gamma;

  // 1/(4*pi*epsilon_0) = 0.89875518e10;

  // For checking purposes the source charge is launched with zero offsets
  // so it becomes, in effect, the reference particle. Then the
  // force per electron (actually positron) test particle are calculated
  // i.e. quantities referred to here as "forces" are actually forces acting
  // on a particle with (positive) charge equal to "e" the charge
  // on a positron. So the units of "force" are [V/m]. 

  // Since px and py are scaled to the nominal beam momentum p_0,
  // the impulse product fx*time has to be divided by p_0, which is done 
  // at this point; this includes factor 10^-9, for GeV units.
  // Also, since a factor "length"
  // will be supplied in "StringSCSolver::propagate", it is convenient
  // to (artificially) divide fx and fy by beta at this point.
  // We assume it is accurate enough to use "beta" (central velocity)
  // for the "velocity" factor.

  // The energy component is also scaled to p_0. But
  //     energy_change = fs*length
  // so no "beta" factor is required for "fs"
  // Also, since the units of "ps" are [GeV], it is necessary to
  // divide "fs" by 10^9.

  m_elemcharge = bunch.getBeamAttributes().getCharge();
  m_macrosize = bunch.getBeamAttributes().getMacrosize();
  double fac = 8.9875518 * m_elemcharge*m_macrosize/(2.0*m_strL)/p_0;

  if( (is==it) || (m_macrosize==0.0) ){
    // No force of string on itself. 
    // NOTE: for Np=1 there is no space charge force.
    // 22 July, 2004, forces have already been initialized to zero
    // 22 July, 2007, skip the force calculation if charge is zero
    return;
  }

  PAC::Position& ps = bunch[is].getPosition();
  PAC::Position& pt = bunch[it].getPosition();

  // Calculate force on test particle due to source particle

  double x = pt.getX() - ps.getX();
  double y = pt.getY() - ps.getY();

  double headSp;
  double tailSp;
  int status;

  TEAPOT::StringSCSolver::Force bodyForce;
  TEAPOT::StringSCSolver::Force endForce;

  TEAPOT::StringSCSolver::Force HPSchForce_h;
  TEAPOT::StringSCSolver::Force HPSchForce_t;
  TEAPOT::StringSCSolver::Force HPSchForce;

  double s_head = pt.getCT() - ps.getCT() - m_strL; 
  double s_tail = pt.getCT() - ps.getCT() + m_strL; 

  double d; 
  double theta_h, theta_t;
  double sintheta_h, costheta_h, den_h; 
  double sintheta_t, costheta_t, den_t; 

  double m_tinier = 1.0e-10;     // must be negligible relative to transverse bunch size

  if ( Ri > 0 ) {
    m_PiByRi = 3.14/Ri; // introduced 24 Jul 2007
                        // no appreciable effect to "npDepend-36mu-ls3-100-seed++"

    status = calculateHeadSp(x, y, s_head, gamma, Ri, sRi, -m_PiByRi, m_PiByRi, headSp);
    if(status != GSL_SUCCESS) {
      std::cerr << "  headSp: " << headSp << "  headSp is not converged (7), set headSp=1.0" << std::endl;
      headSp=1.0;
    }
    status = calculateTailSp(x, y, s_tail, gamma, Ri, sRi, -m_PiByRi, headSp, tailSp);
    if(status != GSL_SUCCESS) {
      std::cerr << "  tailSp: " << tailSp << "  tailSp is not converged (8), set tailSp=-1.0" << std::endl;
      tailSp=-1.0;
    }

    bodyForce = calculateBodyForce0(x, y, Ri, sRi, tailSp, headSp);
    endForce  = calculateEndForce0(x, y, gamma, Ri, sRi, tailSp, headSp);
    m_force[is][it].fx += sRi*fac*(bodyForce.fx + endForce.fx)/beta;
    m_force[is][it].fy += fac*(bodyForce.fy + endForce.fy)/beta;
    m_force[is][it].fs += fac*(bodyForce.fs + endForce.fs);

    bodyForce = calculateBodyForceg2i(x, y, Ri, sRi, tailSp, headSp);
    endForce = calculateEndForceg2i(x, y, gamma, Ri, sRi, tailSp, headSp);
    m_force[is][it].fx += sRi*g2i*fac*(bodyForce.fx + endForce.fx)/beta;
    m_force[is][it].fy += g2i*fac*(bodyForce.fy + endForce.fy)/beta;
    m_force[is][it].fs += g2i*fac*(bodyForce.fs + endForce.fs);
  }
  else {
    // 25 July, 2006
    // Use HPSch = Heaviside, Poincare, Schott formula for force, 
    // Eq. (13.16) of "Accelerator X-Ray Sources" by RT 
    // Integral over string length is in RT notebook.24 July, 2006,
    // and MAPLE program "HeavisidePoinSc.mw
    // Since this gives the electric field it is necessary to include and
    // extra factor of 1/gamma^2 in the transverse force components.
    // This cancels a factor "1/(1-beta^2)" that enters via the transverse 
    // integral over length.
    
    d = sqrt(x*x + (y*y) + m_tinier*m_tinier);
    theta_h = atan2(d, s_head);
    theta_t = atan2(d, s_tail);

    sintheta_h = sin(theta_h);
    costheta_h = cos(theta_h);

    // The transverse force should be repulsive, parallel to (x,y). Check this!
    // assert ( costheta_h < 0 );

    den_h = sqrt(1 - beta*beta*sintheta_h*sintheta_h);
    HPSchForce_h.fx = -g2i*x/d*costheta_h/d/den_h;
    HPSchForce_h.fy = -g2i*y/d*costheta_h/d/den_h;
    HPSchForce_h.fs = g2i*sintheta_h/d/den_h;

    sintheta_t = sin(theta_t);
    costheta_t = cos(theta_t);

    // The transverse force should be repulsive, parallel to (x,y). Check this!
    // assert ( costheta_t < 0 );

    den_t = sqrt(1 - beta*beta*sintheta_t*sintheta_t);
    HPSchForce_t.fx = -g2i*x/d*costheta_t/d/den_t;
    HPSchForce_t.fy = -g2i*y/d*costheta_t/d/den_t;
    HPSchForce_t.fs = g2i*sintheta_t/d/den_t;

    HPSchForce.fx = HPSchForce_h.fx - HPSchForce_t.fx;
    HPSchForce.fy = HPSchForce_h.fy - HPSchForce_t.fy;
    HPSchForce.fs = HPSchForce_h.fs - HPSchForce_t.fs;

    m_force[is][it].fx += fac*HPSchForce.fx;
    m_force[is][it].fy += fac*HPSchForce.fy;
    m_force[is][it].fs += fac*HPSchForce.fs;
  }

    // Factors need to be restored in forcecomp to compare with CSCF paper.
    // In "propagate"
    //        dde = length * m_force[is][it].fs
    // The corresponding longitudinal electric field E_s would satisfy
    //        dde = length * q_el[C]*E_s[V/m] / E_e[eV]
    // where E_e is the electron energy
    //        E_e[Joules] = e * 10^9
    // where e[GeV] is electron energy internal to UAL
    // Hence
    //        E_s(V/m) = e[GeV] * 10^9 * m_force[is][it].fs 

    // In "propagate"
    //        dpx = length[m] * m_force[is][it].fx;
    // The corresponding transverse electric field E_x would satisfy
    //        dpx = (length[m]/(c*beta)) * q_el * E_x[V/m] / p_0[MKS]
    //            = (length[m]/beta) * E_x[V/m] / (cp_0[MKS]/q_el)
    // Hence
    //        E_x[V/m] = beta * m_force[is][it].fx * cp_0[GeV] * 10^9

    // The output appearing in "out/forcecomp" is to be post-processed using
    // "perl/doitall" and the results plotted, 
    // e.g., using "perl/Fs.gnuplot" which assigns forces to coarser bins.

    // The following code should be commented out except when using a line
    // containing a single kick, such as "onekick.mad" (which is the same as
    //  "cellp.mad" but with all kickers except the first removed.

    /*
    char line[120];
    sprintf (line, "%4d %12.7f %12.7f %12.7f %12.3f %12.6f %12.3f %6d %6d",  
      m_counter, pt.getX(), pt.getY(), pt.getCT(), p_0*1.0e9*beta*m_force[is][it].fx, 
      p_0*1.0e9*beta*m_force[is][it].fy, e*1.0e9*m_force[is][it].fs, is, it);
     // forcecomp << line << std::endl;
    std::cout << line << std::endl;
    */
}

void TEAPOT::StringSCSolver::propagate(PAC::Bunch& bunch, double L)
{
  for(int is = 0; is < bunch.size(); is++){
    if(bunch[is].isLost()) continue;
    for(int it = 0; it < bunch.size(); it++){ 
      if(bunch[it].isLost()) continue;
      propagate(bunch, L, is, it);
    }
  }
}

// double x_dde_acc = 0.0;

void TEAPOT::StringSCSolver::propagate(PAC::Bunch& bunch, double length, int is, int it)
{
  // if( it==0 ) return; // TEST PURPOSES ONLY. NO deflection of particle 0

  PAC::Position& pt = bunch[it].getPosition();

  double px = pt.getPX();
  double py = pt.getPY();
  double de = pt.getDE();

  // The source particle has been assigned charge "m_elemcharge*m_macrosize"
  // but, since px, py, and ps are momenta per elementary charge,
  // the test particle charge is the charge on one elementary particle.

  double dpx = length * m_force[is][it].fx;
  double dpy = length * m_force[is][it].fy;
  double dde = length * m_force[is][it].fs;

  px += dpx;  
  py += dpy;
  de += dde;

  pt.setPX(px);
  pt.setPY(py); 
  pt.setDE(de);

  /*
  double x = pt.getX();
  x_dde_acc += x*dde;
  if( (is==0) && (it==1) ){
    std::cout.precision(4);
    std::cout << x_dde_acc << std::endl;
    std::cout.precision(6);
  }
  */
}

// The "0" of "calculateBodyForce0" corresponds to the gamma-independent
// portion, given by Eq.(62).
// Note that "calculateBodyForce0" does not need "gamma" argument

TEAPOT::StringSCSolver::Force  TEAPOT::StringSCSolver::calculateBodyForce0(double x, double y, double Ri, int sRi, double spTail, double spHead)
{
  TEAPOT::StringSCSolver::Force force;

  // double B = -2.0*R*(R + x);
  // double ApB = x*x + (y*y);
  // double A = ApB - B;

  double Bp = -2*(1+x*Ri);
  double AppBp = Ri*Ri*(x*x+(y*y+(m_strH)*(m_strH)));

  // force.fs = -R*R*(IsHead - IsTail); 
  force.fs = -Ri*(calculateIs(AppBp, Bp, Ri, sRi, spHead) - calculateIs(AppBp, Bp, Ri, sRi, spTail));

  // Fx and Fy

  // Check that spHead > spTail and that Lstr is always sufficiently great 
  // that it is impossible for both spTail and spHead to lie in the range 
  // from -m_tiny to m_tiny.

  // assert( spHead > spTail );
  // assert( (spHead-spTail)*(spHead-spTail) > m_tinyByRi*m_tinyByRi );

  // If either lies in range -m_tiny to m_tiny 
  // it will be moved to be on the same side as the other. 

  if( (spHead+m_tinyByRi)*(spHead-m_tinyByRi) < 0 ) spHead = -m_tinyByRi;
  if( (spTail+m_tinyByRi)*(spTail-m_tinyByRi) < 0 ) spTail =  m_tinyByRi;

  // double IxyHead = calculateIxy(ApB, B, aHead);
  // double IxyTail = calculateIxy(ApB, B, aTail);

  // "spHead*spTail" is positive if and only if the origin is outside the
  // range from "spTail" to " spHead"

  if ( spHead*spTail > 0.0 ) {
    // force.fx = R*(2*R + x)*(IxyHead - IxyTail);
    // force.fy = R*y*(IxyHead - IxyTail);
    double tmp = calculateIxy(AppBp, Bp, Ri, sRi, spHead) - calculateIxy(AppBp, Bp, Ri, sRi, spTail);
    force.fx = Ri*(2 + x*Ri)*tmp;
    force.fy = Ri*Ri*y*tmp;
  }
  else
  {
    // double Ixy0p = calculateIxy(ApB, B,  m_tiny);
    // double Ixy0m = calculateIxy(ApB, B, -m_tiny);
    // force.fx = R*(2*R + x)*(IxyHead - Ixy0p + Ixy0m - IxyTail);
    // force.fy = R*y*(IxyHead - Ixy0p + Ixy0m - IxyTail);
    double tmp = calculateIxy(AppBp, Bp, Ri, sRi, spHead) 
               - calculateIxy(AppBp, Bp, Ri, sRi, m_tinyByRi) 
               + calculateIxy(AppBp, Bp, Ri, sRi, -m_tinyByRi)
               - calculateIxy(AppBp, Bp, Ri, sRi, spTail);
    force.fx = Ri*(2 + x*Ri)*tmp;
    force.fy = Ri*Ri*y*tmp;
  }
  return force;
}

// calculateEndForce0 corresponds to gamma-independent part, Eq.(69)

TEAPOT::StringSCSolver::Force  TEAPOT::StringSCSolver::calculateEndForce0(double x, double y, double gamma, double Ri, int sRi, double spTail, double spHead)
{
  TEAPOT::StringSCSolver::Force force;

  double beta = sqrt(gamma*gamma - 1)/gamma;

  // Fs
  double endIsHead = calculateEndIs0(x, y, beta, Ri, sRi, spHead);
  double endIsTail = calculateEndIs0(x, y, beta, Ri, sRi, spTail);

  force.fs = beta*(endIsHead - endIsTail);

  // Fx
  double endIxHead = calculateEndIx0(x, y, beta, Ri, sRi, spHead);
  double endIxTail = calculateEndIx0(x, y, beta, Ri, sRi, spTail);

  force.fx = beta*(endIxHead - endIxTail);

  // Fy
  double endIyHead = calculateEndIy0(x, y, beta, Ri, sRi, spHead);
  double endIyTail = calculateEndIy0(x, y, beta, Ri, sRi, spTail);

  force.fy = beta*(endIyHead - endIyTail);

  return force;
}

// The "g2i" of "calculateBodyForceg2i" corresponds to contribution
// proportional to 1/gamma^2, given by Eq.(63)

TEAPOT::StringSCSolver::Force  TEAPOT::StringSCSolver::calculateBodyForceg2i(double x, double y, double Ri, int sRi, double spTail, double spHead)
{  
  TEAPOT::StringSCSolver::Force force;

  // Fs
  force.fs = 0.0;

  // double B = -2.0*R*(R + x);
  // double ApB = x*x + (y*y);

  double Bp = -2*(1+x*Ri);
  double AppBp = Ri*Ri*(x*x+(y*y+(m_strH)*(m_strH)));

  // Fx and Fy

  // Check that aHead > aTail and that Lstr is always sufficiently great 
  // that it is impossible for both aTail and aHead to lie in the range 
  // from -m_tiny to m_tiny. 

  // assert( spHead > spTail );
  // assert( (spHead-spTail)*(spHead-spTail) > m_tinyByRi*m_tinyByRi );

  // If either lies in range -m_tiny to m_tiny 
  // it will be moved to be on the same side as the other. 

  if( (spHead+m_tinyByRi)*(spHead-m_tinyByRi) < 0 ) spHead = -m_tinyByRi;
  if( (spTail+m_tinyByRi)*(spTail-m_tinyByRi) < 0 ) spTail = m_tinyByRi;

  // double IxyHead = calculateIxy(ApB, B, aHead);
  // double IxyTail = calculateIxy(ApB, B, aTail);
  // double IxyHead = Ri*Ri*Ri*calculateIxy(AppBp, Bp, aHead);
  // double IxyTail = Ri*Ri*Ri*calculateIxy(AppBp, Bp, aTail);

  // std::cout << IxyHead << "   " << Ri*Ri*Ri*calculateIxy(AppBp, Bp, aHead) << "\n"  << std::endl;
 
  // double I0Head = calculateI0(ApB, B, aHead);
  // double I0Tail = calculateI0(ApB, B, aTail);
  // double I0Head = Ri*Ri*Ri*calculateI0(AppBp, Bp, aHead);
  // double I0Tail = Ri*Ri*Ri*calculateI0(AppBp, Bp, aTail);

  // "spHead*spTail" is positive if and only if the origin is outside the
  // range from "spTail" to "spHead"

  if ( spHead*spTail > 0.0 ) {
    force.fx = -Ri*(calculateIxy(AppBp, Bp, Ri, sRi, spHead) 
              - calculateIxy(AppBp, Bp, Ri, sRi, spTail)) 
              + Ri*Ri*x*(calculateI0(AppBp, Bp, Ri, sRi, spHead) 
              - calculateI0(AppBp, Bp, Ri, sRi, spTail));
    force.fy = Ri*Ri*y*(calculateI0(AppBp, Bp, Ri, sRi, spHead) 
                      - calculateI0(AppBp, Bp, Ri, sRi, spTail));
  }
  else
  {
    // double Ixy0p = calculateIxy(ApB, B, m_tiny);
    // double Ixy0m = calculateIxy(ApB, B, -m_tiny);
    // double I00p = calculateI0(ApB, B, m_tiny);
    // double I00m = calculateI0(ApB, B, -m_tiny);
    // double Ixy0p = Ri*Ri*Ri*calculateIxy(AppBp, Bp, m_tiny);
    // double Ixy0m = Ri*Ri*Ri*calculateIxy(AppBp, Bp, -m_tiny);
    // double I00p = Ri*Ri*Ri*calculateI0(AppBp, Bp, m_tiny);
    // double I00m = Ri*Ri*Ri*calculateI0(AppBp, Bp, -m_tiny);

    force.fx = -Ri*(calculateIxy(AppBp, Bp, Ri, sRi, spHead) 
                  - calculateIxy(AppBp, Bp, Ri, sRi, m_tinyByRi) 
                  + calculateIxy(AppBp, Bp, Ri, sRi, -m_tinyByRi) 
                  - calculateIxy(AppBp, Bp, Ri, sRi, spTail)) 
         + Ri*Ri*x*(calculateI0(AppBp, Bp, Ri, sRi, spHead) 
                  - calculateI0(AppBp, Bp, Ri, sRi, m_tinyByRi) 
                  + calculateI0(AppBp, Bp, Ri, sRi, -m_tinyByRi) 
                  - calculateI0(AppBp, Bp, Ri, sRi, spTail));
    force.fy = Ri*Ri*y*(calculateI0(AppBp, Bp, Ri, sRi, spHead) 
                      - calculateI0(AppBp, Bp, Ri, sRi, m_tinyByRi) 
                      + calculateI0(AppBp, Bp, Ri, sRi, -m_tinyByRi) 
                      - calculateI0(AppBp, Bp, Ri, sRi, spTail));
  }
  return force;
}

// The "g2i" of "calculateEndForceg2i" corresponds to contribution
// proportional to 1/gamma^2, given by Eq.(70). Note that there is
// a factor 1/(k*r'^2) missing in that equation in the PRSTAB paper.

TEAPOT::StringSCSolver::Force  TEAPOT::StringSCSolver::calculateEndForceg2i(double x, double y, double gamma, double Ri, int sRi, double spTail, double spHead)
{  
  TEAPOT::StringSCSolver::Force force;

  double beta = sqrt(gamma*gamma - 1)/gamma;

  // Fs
  force.fs = 0.0;

  // Fx
  double endIxHead = calculateEndIxg2i(x, y, beta, Ri, sRi, spHead);
  double endIxTail = calculateEndIxg2i(x, y, beta, Ri, sRi, spTail);

  force.fx = beta*(endIxHead - endIxTail);

  // Fy
  double endIyHead = calculateEndIyg2i(x, y, beta, Ri, sRi, spHead);
  double endIyTail = calculateEndIyg2i(x, y, beta, Ri, sRi, spTail);

  force.fy = beta*(endIyHead - endIyTail);

  return force;
}

// 3 August, 2004. Since A and -B are numerically nearly equal, it seems sensible
// to define a new variable ApB=A+B, and to call all integrals with the
// arguments B and ApB

double TEAPOT::StringSCSolver::calculateIs(double ApB, double B, double Ri, int sRi, double sp)
{
  double sina = sin(sp*Ri/2.0);
  return 2.0/B/sqrt(ApB-2.0*B*sina*sina);
}

double TEAPOT::StringSCSolver::calculateIxy(double ApB, double B, double Ri, int sRi, double sp)
{ 
  double A = ApB-B;
  double cosa = cos(sp*Ri/2.0);
  double sina = sin(sp*Ri/2.0);
  double tmp1 = sqrt(ApB - 2*B*sina*sina);
  double tmp2 = tmp1*sqrt((A-B)*sina*sina);

  gsl_mode_t mode = GSL_PREC_DOUBLE;
  double k        = sqrt(-2.0*B/(A-B));
  double phi      = asin(cosa);

  double F = gsl_sf_ellint_F (phi, k, mode);
  double E = gsl_sf_ellint_E (phi, k, mode);

  double r;

  r  = (F - E)*tmp2;
  r -= B*sin(sp*Ri)*sina;
  r /= tmp1;
  r /= B*(A - B)*sina;
  r *= 2.0;

  return r;
}

double TEAPOT::StringSCSolver::calculateI0(double ApB, double B, double Ri, int sRi, double sp)
{ 
  double A = ApB-B;
  double cosa = cos(sp*Ri/2.0);
  double sina = sin(sp*Ri/2.0);
  double tmp1 = sqrt(ApB - 2*B*sina*sina);
  double tmp2 = sqrt(sina*sina);
  double tmp3 = sqrt(A-B);

  gsl_mode_t mode = GSL_PREC_DOUBLE;
  double k        = sqrt(-2.0*B/(A-B));
  double phi      = asin(cosa);

  double E = gsl_sf_ellint_E (phi, k, mode);

  double r = -E*tmp1*tmp2*tmp3;
  r -= 2*B*cosa*sina*sina;

  r /= tmp1;
  r /= ApB*(A-B);
  r /= sina;
  r *= 2.0;

  double rtmp = TEAPOT::StringSCSolver::calculateIxy(ApB, B, Ri, sRi, sp);
  r -= rtmp;

  return r;
}

// Note: the square root expression for "r" in "calculateEndIs0",
// "calculateEndIx0", etc. is identically equal to the square root expression
// in "headSpEquation" and "tailSpEquation"

// May 31, 2006. Express all end forces and K in terms of 
//    r'/R == rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y)*Ri*Ri)
// in order to eliminate explicit factors of R.
//    1/r'   = Ri/rpByR
//    R/r'^2 = Ri/rpByR^2
// Later sp will be eliminated in favor of sp=sp R

double TEAPOT::StringSCSolver::calculateEndIs0(double x, double y, double beta, double Ri, int sRi, double sp)
{  
  double sina = sin(sp*Ri);
  double cosa = cos(sp*Ri);
  double rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y+(m_strH)*(m_strH))*Ri*Ri);
  double k = 1.0 + beta*(1 + x*Ri)*sina/rpByR;
  return -sina*Ri/rpByR/rpByR/k - beta*cosa*Ri/rpByR/k;
}

double TEAPOT::StringSCSolver::calculateEndIx0(double x, double y, double beta, double Ri, int sRi, double sp)
{ 
  double sina = sin(sp*Ri);
  double cosa = cos(sp*Ri);
  double rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y+(m_strH)*(m_strH))*Ri*Ri);
  double k = 1.0 + beta*(1 + x*Ri)*sina/rpByR;
  return (2+x*Ri)*(1-cosa)*Ri/rpByR/rpByR/k + beta*sina*Ri/rpByR/k;
}

double TEAPOT::StringSCSolver::calculateEndIy0(double x, double y, double beta, double Ri, int sRi, double sp)
{ 
  double sina = sin(sp*Ri);
  double cosa = cos(sp*Ri);
  double rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y+(m_strH)*(m_strH))*Ri*Ri);
  double k = 1.0 + beta*(1 + x*Ri)*sina/rpByR;
  return y*(1-cosa)*Ri*Ri/rpByR/rpByR/k;
}

double TEAPOT::StringSCSolver::calculateEndIxg2i(double x, double y, double beta, double Ri, int sRi, double sp)
{ 
  double sina = sin(sp*Ri);
  double cosa = cos(sp*Ri);
  double rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y+(m_strH)*(m_strH))*Ri*Ri);
  double k = 1.0 + beta*(1 + x*Ri)*sina/rpByR;
  return (-1 + cosa + x*Ri*cosa)*Ri/rpByR/rpByR/k;
}

double TEAPOT::StringSCSolver::calculateEndIyg2i(double x, double y, double beta, double Ri, int sRi, double sp)
{ 
  double sina = sin(sp*Ri);
  double cosa = cos(sp*Ri);
  double rpByR = sqrt(1 + (1+x*Ri)*(1+x*Ri) - 2*(1+x*Ri)*cosa + (y*y+(m_strH)*(m_strH))*Ri*Ri);
  double k = 1.0 + beta*(1 + x*Ri)*sina/rpByR;
  return y*cosa*Ri*Ri/rpByR/rpByR/k;
}

/* // Use of simpleHeadSpEquation results in failure to converge in some cases.
double TEAPOT::StringSCSolver::simpleTailSpEquation(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double s    = p->s;
  double beta = p->beta;

  double result = (R*sp + s) - 2*R*beta*sin(sp/2.0);
  return result;
}
*/

double TEAPOT::StringSCSolver::tailSpEquation(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double s    = p->s;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);

  // double result = R*sp + s + beta*sqrt(R*R + (R + x)*(R + x) + (y*y) - 2*R*(R + x)*cos(sp));
  double result = sp + s + beta*sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );

  return result;
}

double TEAPOT::StringSCSolver::tailSpDerivative(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);
  double j_1 = gsl_sf_bessel_j1(sp*Ri/2);

  double sqrtt = sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );
  double result = 1 + 1/beta/sqrtt*(1+x*Ri)*sp*j_0*(j_0 - 0.5*sp*Ri*j_1);

  // std::cout << result << "   " << result2 << "\n"  << std::endl;

  return result;
}

void TEAPOT::StringSCSolver::tailSpFDF(double sp, void* params, double* f, double* df)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double s    = p->s;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);
  double j_1 = gsl_sf_bessel_j1(sp*Ri/2);

  double sqrtt = sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );

  *f  = sp + s + beta*sqrtt;
  *df = 1 + 1/beta/sqrtt*(1+x*Ri)*sp*j_0*(j_0 - 0.5*sp*Ri*j_1);
}

double TEAPOT::StringSCSolver::headSpEquation(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double s    = p->s;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);

  // double result = R*sp + s + beta*sqrt(R*R + (R + x)*(R + x) + (y*y) - 2*R*(R + x)*cos(sp));
  double result = sp + s + beta*sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );

  return result;
}

double TEAPOT::StringSCSolver::headSpDerivative(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);
  double j_1 = gsl_sf_bessel_j1(sp*Ri/2);

  double sqrtt = sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );

  // double result = R + 0.5*beta*(2*R*(R + x)*sin(sp))/sqrt(R*R + (R + x)*(R + x) + (y*y) - 2*R*(R + x)*cos(sp));
  double result = 1 + 1/beta/sqrtt*(1+x*Ri)*sp*j_0*(j_0 - 0.5*sp*Ri*j_1);

  return result;
}

void TEAPOT::StringSCSolver::headSpFDF(double sp, void* params, double* f, double* df)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double x    = p->x;
  double y    = p->y;
  double s    = p->s;
  double beta = p->beta;

  double j_0 = gsl_sf_bessel_j0(sp*Ri/2);
  double j_1 = gsl_sf_bessel_j1(sp*Ri/2);

  double sqrtt = sqrt( (1+x*Ri)*sp*sp*j_0*j_0 + x*x + (y*y) );

  *f  = sp + s + beta*sqrtt;
  *df = 1 + 1/beta/sqrtt*(1+x*Ri)*sp*j_0*(j_0 - 0.5*sp*Ri*j_1);

  // double tmp = sqrt(R*R + (R + x)*(R + x) + (y*y) - 2*R*(R + x)*cos(sp));
  // *f  = R*sp + s + beta*tmp;
  // *df = R + 0.5*beta*(2*R*(R + x)*sin(sp))/tmp;
}

/* // Use of simpleHeadSpEquation results in failure to converge in some cases.
double TEAPOT::StringSCSolver::simpleHeadSpEquation(double sp, void* params)
{
  struct TEAPOT::StringSCSolver::sp_params *p = 
    (struct TEAPOT::StringSCSolver::sp_params *) params;

  double Ri    = p->Ri;
  double s    = p->s;
  double beta = p->beta;

  double result = (R*sp + s) + 2*R*beta*sin(sp/2.0);
  return result;
}
*/

int TEAPOT::StringSCSolver::calculateTailSp(double x, double y, double s, 
		    double gamma, double Ri, int sRi, double spLo, double spHi, double& result)
{ 
  TEAPOT::StringSCSolver::sp_params params;
  params.Ri    = Ri;
  params.sRi    = sRi;
  params.x    = x;
  params.y    = y;
  params.s    = s;
  params.beta = sqrt(gamma*gamma - 1)/gamma;

  gsl_function gslFunction; 
  gslFunction.function = &TEAPOT::StringSCSolver::tailSpEquation;
  gslFunction.params   = &params;

  // 24 July, 04. Include upper and lower sp limits as arguments
  // to support substrings
  // double spLo = -m_PiByRi;
  // double spHi = +m_PiByRi;

  gsl_root_fsolver_set(p_fRootSolver, &gslFunction, spLo, spHi);

  int status;
  int iter = 0;

  double sp = 0, sp0 = 0;

  do {
      iter++;
      
      status = gsl_root_fsolver_iterate(p_fRootSolver);

      sp    = gsl_root_fsolver_root(p_fRootSolver);      
      spLo  = gsl_root_fsolver_x_lower(p_fRootSolver);
      spHi  = gsl_root_fsolver_x_upper(p_fRootSolver);

      status = gsl_root_test_interval(spLo, spHi, 0, m_epsrel);

      if(status == GSL_SUCCESS) break;

      /*
      std::cout 
	<< iter     << " " 
	<< spLo  << " " 
	<< spHi  << " " 
	<< sp    << " " 
	<< spHi - spLo    << std::endl;
      */
  
  } while (/* status == GSL_CONTINUE && */ iter < m_maxIters);

  if(status == GSL_SUCCESS) {
    result = sp;
    return status;
  }

  // 3 July, 2006. Previous three lines are intended to skip the
  // root polishing stage in most cases. If relative accuracy is not
  // met, perhaps because root is near zero, the root polishing stage
  // is given a chance to find the root to within accuracy
  //           m_frac_strL*m_strL

  // Calculate equations with root polishing.

  gsl_function_fdf gslFDF;

  gslFDF.f      = &TEAPOT::StringSCSolver::tailSpEquation;
  gslFDF.df     = &TEAPOT::StringSCSolver::tailSpDerivative;
  gslFDF.fdf    = &TEAPOT::StringSCSolver::tailSpFDF;
  gslFDF.params = &params;

  gsl_root_fdfsolver_set(p_fdfRootSolver, &gslFDF, sp);

  iter = 0;

  do {
      iter++;
      
      status = gsl_root_fdfsolver_iterate(p_fdfRootSolver);

      sp0   = sp;
      sp    = gsl_root_fdfsolver_root(p_fdfRootSolver);      
      status   = gsl_root_test_delta(sp, sp0, m_frac_strL*m_strL, 0);

      if(status == GSL_SUCCESS) break;

      /*
      std::cout 
	<< iter     << " " 
	<< sp0   << " " 
	<< sp    << std::endl;
      */
  
  } while (/*status == GSL_CONTINUE && */ iter < m_maxIters);

  if(status != GSL_SUCCESS) {
    result = 0.0;
    return status;
  }

  result = sp;
  return status;
}

int TEAPOT::StringSCSolver::calculateHeadSp(double x, double y, double s, 
	  double gamma, double Ri, int sRi, double spLo, double spHi, double& result)
{
  TEAPOT::StringSCSolver::sp_params params;
  params.Ri    = Ri;
  params.sRi    = sRi;
  params.x    = x;
  params.y    = y;
  params.s    = s;
  params.beta = sqrt(gamma*gamma - 1)/gamma;

  // Calculate equations with root bracketing

  gsl_function gslFunction; 
  gslFunction.function = &TEAPOT::StringSCSolver::headSpEquation;
  gslFunction.params   = &params;

  // 24 July, 04. Include upper and lower sp limits as arguments
  // to support substrings

  gsl_root_fsolver_set(p_fRootSolver, &gslFunction, spLo, spHi);

  int status;
  int iter = 0;

  double sp = 0, sp0 = 0;

  do {
      iter++;
      
      status = gsl_root_fsolver_iterate(p_fRootSolver);

      sp    = gsl_root_fsolver_root(p_fRootSolver);      
      spLo  = gsl_root_fsolver_x_lower(p_fRootSolver);
      spHi  = gsl_root_fsolver_x_upper(p_fRootSolver);

      status = gsl_root_test_interval(spLo, spHi, 0, m_epsrel);
      if(status == GSL_SUCCESS) break;

      /*
      std::cout 
	<< iter     << " " 
	<< spLo  << " " 
	<< spHi  << " " 
	<< sp    << " " 
	<< spHi - spLo    << std::endl;
      */
  
  } while (/*status == GSL_CONTINUE && */ iter < m_maxIters);

  if(status == GSL_SUCCESS) {
    result = sp;
    return status;
  }

  // Calculate equations with root polishing

  gsl_function_fdf gslFDF;

  gslFDF.f      = &TEAPOT::StringSCSolver::headSpEquation;
  gslFDF.df     = &TEAPOT::StringSCSolver::headSpDerivative;
  gslFDF.fdf    = &TEAPOT::StringSCSolver::headSpFDF;
  gslFDF.params = &params;

  gsl_root_fdfsolver_set(p_fdfRootSolver, &gslFDF, sp);

  iter = 0;

  do {
      iter++;
      
      status = gsl_root_fdfsolver_iterate(p_fdfRootSolver);

      sp0   = sp;
      sp    = gsl_root_fdfsolver_root(p_fdfRootSolver);      
      status   = gsl_root_test_delta(sp, sp0, m_frac_strL*m_strL, 0);

      if(status == GSL_SUCCESS) break;

      /*
      std::cout 
	<< iter     << " " 
	<< sp0   << " " 
	<< sp    << std::endl;
      */
  
  } while (/*status == GSL_CONTINUE && */ iter < m_maxIters);

  if(status != GSL_SUCCESS) {
    result = 0.0;
    return status;
  }

  result = sp;
  return status;
}
