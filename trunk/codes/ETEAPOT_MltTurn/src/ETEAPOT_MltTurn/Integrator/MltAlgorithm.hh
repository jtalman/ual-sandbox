#ifndef ETEAPOT_MLT_ALGORITHM_MLT_TURN_HH
#define ETEAPOT_MLT_ALGORITHM_MLT_TURN_HH

#include <iomanip>
//#include "Main/Teapot.h"
//#include "UAL/UI/OpticsCalculator.hh"
#include "SMF/PacElemMultipole.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/CommonAlgorithm.hh"
#include "ETEAPOT_MltTurn/Integrator/Matrices.hh"

namespace ETEAPOT_MltTurn {
 
  /** A template of the common methods used by the mlt conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class MltAlgorithm
    : public ETEAPOT::CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    MltAlgorithm();

    /** Destructor */
    ~MltAlgorithm();

    /** Passes the element entry  */
    void passEntry(int ip, const ETEAPOT::MltData& edata, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba);

    /** Passes the element exit  */
    void passExit(int ip, const ETEAPOT::MltData& edata, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba);

   /** Applies the multipole kick */
   void applyMltKick(int ip, const ETEAPOT::MltData& edata, double rkicks, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p, int mltK, double m_m);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p, double m_m);
// void applyMltKick(const MltData& edata, double rkicks, Coordinates& p);

//  static std::string Mlt_m_sxfFilename;
    static std::string Mlt_m_elementName[1000];
    static double Mlt_m_sX[1000];
    static double spin[41][3];

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(int ip, PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, int mltK, double m_m);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p, double m_m);
//  void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);
//

  };

}

#include "ETEAPOT_MltTurn/Integrator/MltAlgorithm.icc"

#endif
