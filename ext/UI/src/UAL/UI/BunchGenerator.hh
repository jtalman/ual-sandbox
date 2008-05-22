#ifndef UAL_BUNCH_GENERATOR_HH
#define UAL_BUNCH_GENERATOR_HH

#include <string>

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTwissData.h"
#include "UAL/UI/Arguments.hh"

namespace UAL {

  /** UI Bunch Generator */


  class BunchGenerator {

  public:

    /** Constructor */
    BunchGenerator();

    /** Defines bunch arguments */
    bool setBunchArguments(const UAL::Arguments& arguments);

    /** Generate bunch distribution */
    void updateBunch(PAC::Bunch& bunch, PacTwissData& twiss);

    int getBunchSize() { return m_np; }

    const std::string& getType() { return m_type; }
    void setType(std::string& type) { m_type = type; }

    double getEnX() { return m_enx; }
    void setEnX(double enx) { m_enx = enx; }

    double getEnY() { return m_eny; }
    void setEnY(double eny) { m_eny = eny; }

  public:

    double ctHalfWidth, deHalfWidth; 

  protected:

    void updateGaussBunch(PAC::Bunch& bunch, PacTwissData& twiss,
			  double v0byc, double gamma);
    void updateGridBunch(PAC::Bunch& bunch, PacTwissData& twiss,
			 double v0byc, double gamma);

  protected:

    std::string m_type;

    int m_np;
    double m_enx, m_eny, m_et;
    int m_seed;

  };


}

#endif
