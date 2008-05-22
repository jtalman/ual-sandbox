// Library       : UAL
// File          : UAL/JVM/JWindowManager.hh
// Copyright     : see Copyright file
// Author        : N.Malitsky

#ifndef UAL_JWINDOW_MANAGER_HH
#define UAL_JWINDOW_MANAGER_HH

#include "UAL/JVM/JavaSingleton.hh"

namespace UAL {

  /** C++ proxy of the Java Window Manager */

  class JWindowManager : public Object {

  public:

    /** Constructor */
    JWindowManager();

    /** Destructor */
    ~JWindowManager();

    /** Initialize the Main Window */
    void initMainWindow();

    /** Shows the Main Window */
    void showMainWindow();

    /** Hides the Main Window */
    void hideMainWindow();

  protected:

    /** Global reference to the Java class */
    jobject  m_jclass;

    /** Global reference to the Java object */
    jobject m_jobject;

  private:

    // method Ids
    jmethodID m_initMainWindow_id;
    jmethodID m_showMainWindow_id;
    jmethodID m_hideMainWindow_id;
     
  };

};

#endif
