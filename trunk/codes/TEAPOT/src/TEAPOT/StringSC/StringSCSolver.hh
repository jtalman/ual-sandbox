// Library       : TEAPOT
// File          : TEAPOT/StringSC/StringSCSolver.hh
// Copyright     : see Copyright file
// Author        : R.Talman

#ifndef UAL_TEAPOT_STRING_SC_SOLVER_HH
#define UAL_TEAPOT_STRING_SC_SOLVER_HH

#include <string>
#include <vector>
#include <map>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_bessel.h>

#include "UAL/Common/Def.hh"
#include "PAC/Beam/Bunch.hh"

namespace TEAPOT 
{

  class StringSCSolver 
  {

  public:

    /** Returns a singleton */
    static StringSCSolver& getInstance();

    /** Destructor */
    ~StringSCSolver();

    /* 3 June, 2006. Inverse bend radius Ri is retained as positive value Ri=|Ri|
       and sign sRi. Note though that sRi=1 when Ri=0. */

    /** Calculate force */
    void calculateForce(PAC::Bunch& bunch, double Ri, int sRi);
    void calculateForce(PAC::Bunch& bunch, double Ri, int sRi, int is, int it); // should be protected

    /** Propagate bunch (inherited from the base class */
    void propagate(PAC::Bunch& bunch);

    // should be protected
    void propagate(PAC::Bunch& bunch, double Ri, int sRi, double T);
    void propagate(PAC::Bunch& bunch, double L);
    void propagate(PAC::Bunch& bunch, double L, int is, int it );

    void setCounter(int counter);
    void setMaxIters(int maxIters);
    void setMaxBunchSize(int np);

    void setStringL(double strL);
    void setStringH(double yhH);

    /** New Setters and getters */

    void setL(const std::string& name, double L)         { m_Ls[name] = L;}
    double getL(const std::string& name)                 { return m_Ls[name]; }
    void setAngle(const std::string& name, double angle) { m_angles[name] = angle;}
    double getAngle(const std::string& name)             { return m_angles[name]; }


    /** Setters and getters */
    /* 
    3 June, 2006. Inverse bend radius Ri is retained as positive value Ri=|Ri|
    and the bend orientation is maintained by (sign) sRi. Note though that 
    sRi=1 when Ri=0. Bend angles, with their correct algebraic signs, are entered  
    via instructions such as 
              scSolver.setAngle("k2", -0.0163625);
    in "main.cc".
    */

    void setL(double L) { m_L = L; }     // L is length of bending element
    double getL()       { return m_L; }
    void setRi(double Ri) { m_Ri = Ri; }
    double getRi()       { return m_Ri; }

    void setBendfac(double bendfac) { m_bendfac = bendfac; }
    double getBendfac() { return m_bendfac; }

    void setElemcharge(double ELEMCHARGE) { m_elemcharge = ELEMCHARGE; }
    double getElemcharge()       { return m_elemcharge; } 
    void setMacrosize(double MACROSIZE) { m_macrosize = MACROSIZE; }
    double getMacrosize()       { return m_macrosize; }

  protected:

    std::map<std::string, double> m_Ls;
    std::map<std::string, double> m_angles;

  private:

    /** Constructor */
    StringSCSolver();

  private:

    struct sp_params
    {
      double Ri;
      int sRi;
      double x;
      double y;
      double s;
      double beta;
    };

    struct Force 
    {
      double fx;
      double fy;
      double fs;
    };

  private:

    double m_L;
    double m_Ri;
    int m_sRi;

    double m_bendfac; 

  private:

    double m_tiny;
    double m_tinyByRi;
    double m_PiByRi;
    double m_epsrel;
    double m_frac_strL;
    int    m_maxIters;
    double m_strL;
    double m_strH;

  private:

    int m_counter;
    std::vector< std::vector<Force> > m_force;

    double m_elemcharge;
    double m_macrosize;

  private:
    // 24 July, 04. Include upper and lower sp limits to support substrings
    int calculateTailSp(double x, double y, double s, double gamma, double Ri, int sRi, double spLo, double spHi, double& result);
    int calculateHeadSp(double x, double y, double s, double gamma, double Ri, int sRi, double spLo, double spHi, double& result);

    Force calculateBodyForce0(double x, double y, double Ri, int sRi, double aTail, double aHead);
    Force calculateEndForce0(double x, double y, double gamma, double Ri, int sRi, double aTail, double aHead);

    Force calculateBodyForceg2i(double x, double y, double Ri, int sRi, double aTail, double aHead);
    Force calculateEndForceg2i(double x, double y, double gamma, double Ri, int sRi, double aTail, double aHead);

  public:

    static double simpleHeadSpEquation(double sp, void* params); 
    static double headSpEquation(double sp, void* params);
    static double headSpDerivative(double sp, void* params);
    static void   headSpFDF(double sp, void* params, double *y, double *dy); 

    static double simpleTailSpEquation(double sp, void* params); 
    static double tailSpEquation(double sp, void* params);
    static double tailSpDerivative(double sp, void* params);
    static void   tailSpFDF(double sp, void* params, double *y, double *dy); 

  private: // will be replaced with private

    double calculateIs(double ApB, double B, double Ri, int sRi, double sp);
    double calculateIxy(double ApB, double B, double Ri, int sRi, double sp);
    double calculateI0(double ApB, double B, double Ri, int sRi, double sp);

    double calculateEndIs0(double x, double y, double beta, double Ri, int sRi, double sp);
    double calculateEndIx0(double x, double y, double beta, double Ri, int sRi, double sp);
    double calculateEndIy0(double x, double y, double beta, double Ri, int sRi, double sp);

    double calculateEndIxg2i(double x, double y, double beta, double Ri, int sRi, double sp);
    double calculateEndIyg2i(double x, double y, double beta, double Ri, int sRi, double sp);

  private:

    static StringSCSolver* s_theInstance;

  private:

    gsl_root_fsolver* p_fRootSolver; 
    gsl_root_fdfsolver* p_fdfRootSolver; 

  };

};


#endif

