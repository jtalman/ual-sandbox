#ifndef UAL_ROOT_BUNCH_TBT_VIEWER_HH
#define UAL_ROOT_BUNCH_TBT_VIEWER_HH

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

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{

 namespace ROOT {

  class BunchTBTViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BunchTBTViewer(UAL::QT::BasicPlayer* player);

    /** Updates plot */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** Updates points */
    void updatePoints(int turn);

    /** Gets a turn from bpm data */
    int getTurn();

  private:

    // Parent player
    UAL::QT::BasicPlayer* p_player; 

    int m_bpmIndex;

    TGraph* xTBT;
    TGraph* yTBT;
    TGraph* ctTBT;

  private:

  };
 }
}

#endif
