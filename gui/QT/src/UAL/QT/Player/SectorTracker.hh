#ifndef UAL_RHIC_SECTOR_TRACKER_HH
#define UAL_RHIC_SECTOR_TRACKER_HH

#include "PAC/Beam/Bunch.hh"
#include "UAL/UI/OpticsCalculator.hh"

namespace UAL_RHIC 
{
  class SectorTracker
  {

  public:

    /** Constructor */
    SectorTracker();

    /** Sets optics */
    void setOptics(UAL::OpticsCalculator& optics);

    void propagate(PAC::Bunch& bunch);

  protected:

    double m_suml, m_alpha0;

    double m_mux, m_betax, m_alphax, m_dx, m_dpx;
    double m_muy, m_betay, m_alphay, m_dy, m_dpy;

    double m_chromx;
    double m_chromy;

    double m_dmux60;
    double m_dmuy60;


  };

}


#endif
