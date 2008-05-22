#ifndef UAL_ROOT_BUNCH_Y_PY_VIEWER_HH
#define UAL_ROOT_BUNCH_Y_PY_VIEWER_HH

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
#include "TH2F.h"
#include "TGraph.h"

#include "PAC/Beam/Bunch.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
  namespace ROOT {

    /** Viewer of the bunch distribution on the Y-PY plane */
  class BunchYPyViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BunchYPyViewer(UAL::QT::BasicPlayer* player, PAC::Bunch* bunch);

    /** Processes the event issued by tracker */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** Updates points */
    void updatePoints();

  protected:

    /** pointer to the application's main window */
    UAL::QT::BasicPlayer* p_player;

    /** pointer to a bunch tracked by the UAL model */
    PAC::Bunch* p_bunch;

  private:

    // y-py  plot

    double m_yMax, m_pyMax;
    int m_ybins, m_pybins;

    TH2F* ypy;
   
  private:

    void findLimits();
  };
 }
}

#endif
