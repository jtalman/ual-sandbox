// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleData.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DIPOLE_DATA_HH
#define UAL_TEAPOT_DIPOLE_DATA_HH

#include "SMF/PacLattElement.h"
#include "Survey/PacSurveyData.h"
#include "TEAPOT/Integrator/ElemSlice.hh"

namespace TEAPOT {

  /** Collection of attributes used by the TEAPOT dipole propagator */

  class DipoleData {

  public:

    /** Constructor */
    DipoleData();

    /** Copy constructor */
    DipoleData(const DipoleData& data);

    /** Destructor */
    ~DipoleData();

    /** Copy operator */
    const DipoleData& operator=(const DipoleData& data);

    /** Sets the lattice element */
    void setLatticeElement(const PacLattElement& e);

  protected:

    // /** Set bend angle and strengths */
    // void setBend(double angle);

    /** Set bend strengths */
    void setBendStrengths(const std::string& etype);

  public: 

    /** Sets frame slices*/
    void setSlices(double l, double angle, int ir);
    
    /** Sets frame slices*/
    void setSlices(PacSurveyData& survey, double l, double angle, int ir, int flag);    


  public:

    /** Element length */

    double m_l;

    /** Complexity number */

    double m_ir;

    /** Bend angle */

    double m_angle;

    /** Thin dipole strengths */

    double m_atw00;
    double m_atw01;
    double m_btw00;
    double m_btw01;

    /**  Frame slices */

    std::vector<ElemSlice> m_slices;   

  private:

    void initialize();
    void copy(const DipoleData& data);

  };

}

#endif
