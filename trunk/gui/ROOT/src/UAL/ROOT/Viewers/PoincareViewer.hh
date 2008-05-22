#ifndef UAL_ROOT_POINCARE_VIEWER_HH
#define UAL_ROOT_POINCARE_VIEWER_HH

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
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{

  namespace ROOT {

  class PoincareViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    PoincareViewer(UAL::QT::BasicPlayer* player);

    /** Updates plot */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    // Parent player
     UAL::QT::BasicPlayer* p_player; 

    int m_prevTurn;
    int m_bpmIndex;
    int m_points;

    TGraph* m_tinyGraph; // ad hoc solution to support axis
    vector<TGraph*> m_graphs;

  protected:

    /** Extends a file menu with the "Write To" action */
    void updateMenu();

    /** Writes twiss data into the specified file */
    void writeToFile(const QString& fileName);

    /** Action which is responsible for writing to a file */
    QAction *writeToAct; 

  private slots:

    /** Qt slot associated with writeToAct action */
    bool writeTo(); 
    

  };
 }
}

#endif
