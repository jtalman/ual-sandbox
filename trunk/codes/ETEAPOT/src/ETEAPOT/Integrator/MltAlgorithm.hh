// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltAlgorithm.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_MLT_ALGORITHM_HH
#define ETEAPOT_MLT_ALGORITHM_HH

#include <iomanip>
//#include "Main/Teapot.h"
//#include "UAL/UI/OpticsCalculator.hh"
#include "SMF/PacElemMultipole.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/CommonAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the common methods used by the mlt conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class MltAlgorithm
    : public CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    MltAlgorithm();

    /** Destructor */
    ~MltAlgorithm();

    /** Passes the element entry  */
    void passEntry(int ip, const MltData& edata, Coordinates& p, int mltK, double m_m );

    /** Passes the element exit  */
    void passExit(int ip, const MltData& edata, Coordinates& p, int mltK, double m_m );

   /** Applies the multipole kick */
   void applyMltKick(int ip, const MltData& edata, double rkicks, Coordinates& p, int mltK, double m_m);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p, int mltK, double m_m);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p, double m_m);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p);

//  static std::string Mlt_m_sxfFilename;
    static std::string Mlt_m_elementName[2000];
    static double Mlt_m_sX[2000];

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(int ip, PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, int mltK, double m_m);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, int mltK, double m_m);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, double m_m);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);
//

  };

}

#include "ETEAPOT/Integrator/MltAlgorithm.icc"

#endif
