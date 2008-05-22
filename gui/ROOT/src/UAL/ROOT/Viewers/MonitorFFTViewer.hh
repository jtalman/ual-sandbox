#ifndef UAL_ROOT_MONITOR_FFT_VIEWER_HH
#define UAL_ROOT_MONITOR_FFT_VIEWER_HH

#include <vector>

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TPad.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TQtWidget.h"
#include "TGraph.h"

#include "PAC/Beam/Position.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
  namespace ROOT {

  class MonitorFFTViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    MonitorFFTViewer(UAL::QT::BasicPlayer* player);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** Updates points */
    void updatePoints();

  private:

    // Parent player
    UAL::QT::BasicPlayer* p_player; 

    int m_bpmIndex;

    TGraph* xTunes;
    TGraph* yTunes;

  };
 }
}

#endif
