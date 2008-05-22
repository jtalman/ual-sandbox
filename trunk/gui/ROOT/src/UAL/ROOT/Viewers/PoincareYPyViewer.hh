#ifndef UAL_ROOT_POINCARE_YPY_VIEWER_HH
#define UAL_ROOT_POINCARE_YPY_VIEWER_HH

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
#include "TMultiGraph.h"
#include "TGraph.h"

#include "Optics/PacTwissData.h"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/PoincareViewer.hh"

namespace UAL
{

  namespace ROOT {

  class PoincareYPyViewer : public PoincareViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    PoincareYPyViewer(UAL::QT::BasicPlayer* player);

    /** Updates plot */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  private:

    void initPoints();
    void updatePoints(int turn);

  private:

    /*
    // Parent player
     UAL::QT::BasicPlayer* p_player; 

    int m_prevTurn;
    int m_bpmIndex;
    int m_points;

    TGraph* m_tinyGraph; // ad hoc solution to support axis
    vector<TGraph*> m_graphs;
    */

  private:

  };
 }
}

#endif
