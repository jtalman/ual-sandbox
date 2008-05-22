#ifndef UAL_SEPARATRIX_CALCULATOR_HH
#define UAL_SEPARATRIX_CALCULATOR_HH

#include <vector>

#include "PAC/Beam/BeamAttributes.hh"
#include "TIBETAN/Propagator/RFCavityTracker.hh"

namespace UAL {

  class SeparatrixCalculator {

  public:

    static SeparatrixCalculator& getInstance();

    /** Sets beam attributes */
    void setBeamAttributes(const PAC::BeamAttributes& ba);

    /** Sets lattice */
    void setLattice(double suml, double alpha0);

    /** Sets the RF cavity */
    void setRFCavity(double v, double h, double lag);

    /** Returns RF cavity */
    TIBETAN::RFCavityTracker& getRFCavity() { return m_rfTracker; }

    /** Calculate separatrix */
    bool calculate();

    double getDeMax();
    double getSumL() { return m_suml; }

  public:

    std::vector<double> m_phases;
    std::vector<double> m_des;

  protected:

    /** Beam attributes */
    PAC::BeamAttributes m_ba;

    /** Alpha */
    double m_alpha0;

    /** Accelerator length */  
    double m_suml;

    /** RF */
    TIBETAN::RFCavityTracker m_rfTracker;

  protected:

    double getDeMax(double et);
    double getCtMax(double et);

  private:

    bool   isNearTransition();
    double getHsep(double phase);
    double getHsep(double phase, double v, double harmon, double lag);
    double getV(double phase);
    double getV(double phase, double v, double h, double lag);

  private:

    static SeparatrixCalculator* s_theInstance;

    /** Constructor */
    SeparatrixCalculator();


  };

}

#endif
