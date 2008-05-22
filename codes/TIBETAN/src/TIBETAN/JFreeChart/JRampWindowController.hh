// Library       : TIBETAN
// File          : TIBETAN/JFreeChart/JRampWindowController.hh
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#ifndef UAL_TIBETAN_JRAMP_WINDOW_CONTROLLER_HH
#define UAL_TIBETAN_JRAMP_WINDOW_CONTROLLER_HH

#include <jni.h>
#include "TIBETAN/JFreeChart/BasicJFreeChartProxy.hh"

namespace TIBETAN {

  /** C++ proxy of the Java Ramp Plot Controller */

  class JRampWindowController  : public BasicJFreeChartProxy {

  public:

    /** Constructor */
    JRampWindowController();

    /** Destructor */
    ~JRampWindowController();

    /** Initializes the ramp plot */
    void initWindow();

    /** Shows the ramp plot */
    void showWindow();

    /** Updates plot */
    void updateData();

    /** Sets the range of the time axis */
    void setTimeRange(double minTime, double maxTime);

    /** Sets the range of the gamma axis */
    void setGammaRange(double minGamma, double maxGamma);

    /** Sets the range of the RF axis */
    void setRFRange(double minValue, double maxValue);

    /** Adds the gamma values */
    void addGammaValue(double t, double gamma);


  protected:

    /**  Global reference to the Java class */
    jobject  m_jclass;

    /** Global reference to the Java object */
    jobject m_jobject;

  private:

    // method Ids
    jmethodID m_initWindow_id;
    jmethodID m_showWindow_id;
    jmethodID m_updateData_id;
    jmethodID m_setTimeRange_id;
    jmethodID m_setGammaRange_id;
    jmethodID m_addGammaValue_id;
 
  };


};

#endif
